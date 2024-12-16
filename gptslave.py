from huggingface_hub import InferenceClient
import subprocess
import platform
import sys
import os
import re
import time
import json
import asyncio
import shlex
import pwd
import getpass
import signal
import shutil
from typing import Optional, Dict, Any, List, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

class RateLimitHandler:
    def __init__(self):
        self.last_request = 0
        self.min_delay = 1.0  # Minimum delay between requests
        self.backoff_factor = 2  # Multiplicative factor for backoff
        self.max_delay = 30  # Maximum delay in seconds
        self.current_delay = self.min_delay
        self.error_count = 0
        self.max_retries = 5

    def reset(self):
        self.current_delay = self.min_delay
        self.error_count = 0

    def wait(self):
        now = time.time()
        elapsed = now - self.last_request
        if elapsed < self.current_delay:
            time.sleep(self.current_delay - elapsed)
        self.last_request = time.time()

    def handle_error(self) -> bool:
        """Handle rate limit error. Returns True if should retry."""
        self.error_count += 1
        if self.error_count > self.max_retries:
            return False
        self.current_delay = min(self.current_delay * self.backoff_factor, self.max_delay)
        print(f"\nRate limit hit. Waiting {self.current_delay:.1f} seconds before retry...")
        time.sleep(self.current_delay)
        return True

class CommandResult:
    def __init__(self, command: str, returncode: int, stdout: str, stderr: str):
        self.command = command
        self.returncode = returncode
        self.stdout = stdout[:1000] if stdout else ""
        self.stderr = stderr[:1000] if stderr else ""
        self.timestamp = time.time()

    def __str__(self):
        return f"Command: {self.command}\nReturn code: {self.returncode}\nOutput: {self.stdout}\nError: {self.stderr}"

    def was_successful(self) -> bool:
        return self.returncode == 0

class AIComputerAssistant:
    def __init__(self):
        self.client = InferenceClient(api_key="PROVIDE YOUR KEY FROM HUGGING FACE")
        self.os_type = platform.system().lower()
        self.user = getpass.getuser()
        self.home = os.path.expanduser('~')
        self.available_commands = self._get_available_commands()
        self.command_history: List[CommandResult] = []
        self.task_stack = []
        self.max_retries = 3
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.analysis_timeout = 10
        self.rate_limiter = RateLimitHandler()
        self.backup_dir = os.path.join(os.path.dirname(__file__), 'backups')
        os.makedirs(self.backup_dir, exist_ok=True)
        signal.signal(signal.SIGINT, self.signal_handler)

    def _get_available_commands(self) -> set:
        commands = set()
        paths = os.environ.get('PATH', '').split(os.pathsep)
        for path in paths:
            if os.path.exists(path):
                for cmd in os.listdir(path):
                    cmd_path = os.path.join(path, cmd)
                    if os.path.isfile(cmd_path) and os.access(cmd_path, os.X_OK):
                        commands.add(cmd)
        return commands

    def create_backup(self) -> str:
        """Create a backup of the current script"""
        current_file = os.path.abspath(__file__)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = os.path.join(self.backup_dir, f'agent_backup_{timestamp}.py')
        
        try:
            shutil.copy2(current_file, backup_file)
            print(f"\nBackup created: {backup_file}")
            return backup_file
        except Exception as e:
            print(f"\nError creating backup: {e}")
            return ""

    def restore_backup(self, backup_file: str) -> bool:
        """Restore from a backup file"""
        try:
            current_file = os.path.abspath(__file__)
            shutil.copy2(backup_file, current_file)
            print(f"\nRestored from backup: {backup_file}")
            return True
        except Exception as e:
            print(f"\nError restoring backup: {e}")
            return False

    def signal_handler(self, signum, frame):
        print("\n\nGracefully handling interrupt...")
        if self.task_stack:
            print("Current task stack:", self.task_stack)
        print("Type 'continue' to resume, 'skip' to move to next task, or 'exit' to quit")

    def validate_path(self, path: str) -> tuple[bool, str]:
        """Validate file path for security."""
        try:
            full_path = os.path.realpath(os.path.expanduser(path))
            suspicious_patterns = [
                '/etc/', '/var/', '/usr/', '/bin/', 
                'passwd','sudoers',
                '.ssh/', 
            ]
            
            for pattern in suspicious_patterns:
                if pattern in full_path:
                    return False, f"Access to {pattern} is restricted"
            
            return True, full_path
        except Exception as e:
            return False, str(e)

    async def analyze_command_failure(self, result: CommandResult) -> str:
        """Analyze command failure and suggest fixes"""
        analysis_prompt = f"""
Analyze this failed command and suggest fix:
Command: {result.command}
Return code: {result.returncode}
Error: {result.stderr}
Output: {result.stdout}
"""
        retries = 0
        max_retries = 3
        
        while retries < max_retries:
            try:
                self.rate_limiter.wait()
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model="Qwen/Qwen2.5-72B-Instruct",
                        messages=[
                            {"role": "system", "content": "You are a system expert. Be brief and direct."},
                            {"role": "user", "content": analysis_prompt}
                        ],
                        temperature=0.5,
                        max_tokens=200,
                        stream=False
                    ),
                    timeout=self.analysis_timeout
                )
                self.rate_limiter.reset()
                return response.choices[0].message.content
            except asyncio.TimeoutError:
                print("\nAnalysis timed out. Retrying...")
                retries += 1
            except Exception as e:
                if "429" in str(e):  # Rate limit error
                    if not self.rate_limiter.handle_error():
                        return "Rate limit reached. Skipping analysis."
                    continue
                print(f"\nError in analysis: {e}")
                retries += 1
        
        return "Analysis failed after multiple retries."

    async def execute_system_command(self, command: str, retry_count: int = 0) -> CommandResult:
        """Execute a system command with heredoc support"""
        try:
            if any(dangerous in command.lower() for dangerous in [
                'rm -rf /','mkfs', 'dd if=/dev/zero', 
                '>/dev/sda', ':(){:|:&};:', '> /dev/sda',
               'mv /* /dev/null'
            ]):
                raise ValueError("Potentially dangerous command detected")

            if '<<' in command:
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
            else:
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

            stdout_data = []
            stderr_data = []

            async def read_stream(stream, output_list):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    try:
                        decoded_line = line.decode('utf-8')
                        print(decoded_line.rstrip())
                        output_list.append(decoded_line)
                    except UnicodeDecodeError:
                        decoded_line = line.decode('utf-8', errors='replace')
                        print(decoded_line.rstrip())
                        output_list.append(decoded_line)

            await asyncio.gather(
                read_stream(process.stdout, stdout_data),
                read_stream(process.stderr, stderr_data)
            )

            returncode = await process.wait()

            result = CommandResult(
                command=command,
                returncode=returncode,
                stdout=''.join(stdout_data),
                stderr=''.join(stderr_data)
            )

            if not result.was_successful() and retry_count < self.max_retries:
                print(f"\n✗ Command failed (attempt {retry_count + 1}/{self.max_retries + 1})")
                
                print("\nAnalyzing what went wrong...")
                analysis = await self.analyze_command_failure(result)
                print(f"\nAnalysis:\n{analysis}")
                
                if input("\nRetry with fixes? (y/n): ").lower() == 'y':
                    fixed_cmd_response = await self.get_response(
                        f"Based on the analysis, provide only the corrected command with SYSTEM: prefix",
                        [str(result)]
                    )
                    
                    if fixed_cmd_response and 'SYSTEM:' in fixed_cmd_response:
                        new_command = fixed_cmd_response.split('SYSTEM:', 1)[1].strip()
                        print(f"\nRetrying with: {new_command}")
                        return await self.execute_system_command(new_command, retry_count + 1)

            return result

        except Exception as e:
            return CommandResult(command, 1, '', str(e))

    def _get_truncated_history(self, max_items: int = 3) -> List[str]:
        """Get truncated command history to avoid token limits"""
        return [str(result) for result in self.command_history[-max_items:] if result.command and (result.stdout or result.stderr)]

    async def get_response(self, user_input: str, context_history: List[str] = None) -> Optional[str]:
        if not user_input.strip():
            return None

        system_message = f"""You are a helpful computer control assistant.
Current user: {self.user}
OS: {self.os_type}
Home: {self.home}

For creating scripts or files with multiple lines, use heredoc format:
SYSTEM:cat > filename.py << 'EOF'
#!/usr/bin/env python3
[content here]
EOF

For other commands use:
SYSTEM:command args

Keep responses focused. For scripts use correct indentation, include imports, add proper error handling and make executable with shebang"""

        messages = [
            {"role": "system", "content": system_message},
        ]

        if context_history:
            truncated_history = context_history[-3:] if len(context_history) > 3 else context_history
            for msg in truncated_history:
                messages.append({"role": "assistant", "content": msg[:1000]})

        messages.append({"role": "user", "content": user_input})

        retries = 0
        max_retries = 3
        while retries < max_retries:
            try:
                self.rate_limiter.wait()
                stream = self.client.chat.completions.create(
                    model="Qwen/Qwen2.5-72B-Instruct",
                    messages=messages,
                    temperature=0.5,
                    max_tokens=2048,
                    top_p=0.7,
                    stream=True
                )

                result = ""
                for chunk in stream:
                    content = chunk.choices[0].delta.content or ""
                    result += content
                    print(content, end='', flush=True)

                self.rate_limiter.reset()
                return result

            except Exception as e:
                if "429" in str(e):  # Rate limit error
                    if not self.rate_limiter.handle_error():
                        print("\nRate limit reached. Please try again later.")
                        return None
                    continue
                print(f"\nError in API communication: {str(e)}")
                retries += 1
                if retries >= max_retries:
                    return None

    async def parse_and_execute_commands(self, response: Optional[str]) -> bool:
        if not response:
            return False
        
        command_pattern = r'SYSTEM:(.+?)(?=SYSTEM:|$)'
        matches = re.findall(command_pattern, response, re.DOTALL | re.MULTILINE)
        
        if not matches:
            print(response)
            return True

        total_commands = len(matches)
        print(f"\nFound {total_commands} commands to execute")
        
        success = True
        for idx, command in enumerate(matches, 1):
            command = command.strip()
            if not command:
                continue
            
            print(f"\n[{idx}/{total_commands}] Executing: {command}")
            
            result = await self.execute_system_command(command)
            self.command_history.append(result)

            if not result.was_successful():
                print(f"✗ Command failed with return code {result.returncode}")
                if result.stderr:
                    print(f"Error output:\n{result.stderr}")
                success = False
                break
            else:
                print("✓ Command completed successfully")

        return success

    async def execute_task(self, task_description: str) -> bool:
        print(f"\nExecuting task: {task_description}")
        self.task_stack.append(task_description)
        
        try:
            print("\nPlanning task execution...")
            response = await self.get_response(task_description, self._get_truncated_history())
            
            if not response:
                print("\nFailed to get response from API. Task aborted.")
                self.task_stack.pop()
                return False

            success = await self.parse_and_execute_commands(response)
            
            while success:
                print("\nChecking for additional steps...")
                response = await self.get_response("Are there any remaining steps? If yes, execute them.", 
                                                self._get_truncated_history())
                
                if not response:
                    print("\nFailed to check for additional steps. Task completed with warning.")
                    break
                
                if not re.search(r'(?:SYSTEM):', response):
                    print("\nTask completed successfully!")
                    break
                    
                success = await self.parse_and_execute_commands(response)
            
            self.task_stack.pop()
            return success
            
        except Exception as e:
            print(f"\nError during task execution: {str(e)}")
            self.task_stack.pop()
            return False

    async def modify_self(self, modification_prompt: str) -> bool:
        """Modify the assistant's own code based on the prompt"""
        print("\nPreparing for self-modification...")
        
        # Create backup first
        backup_file = self.create_backup()
        if not backup_file:
            return False

        try:
            # Read current code
            with open(__file__, 'r') as f:
                current_code = f.read()

            # Prepare the modification prompt
            system_message = f"""You are a Python expert. Modify the following code according to the user's request.
Keep all existing functionality intact and only add/modify what's specifically requested.
Maintain the same code structure and style.
Return only the complete modified code without any explanations.
The code must include all imports and the final asyncio.run(main()) call.
Do not remove or alter any existing functionality.

Current code:
{current_code}

Modification request:
{modification_prompt}
"""

            print("\nGenerating modified code...")
            retries = 0
            max_retries = 3
            while retries < max_retries:
                try:
                    self.rate_limiter.wait()
                    response = await self.client.chat.completions.create(
                        model="Qwen/Qwen2.5-72B-Instruct",
                        messages=[
                            {"role": "system", "content": system_message}
                        ],
                        temperature=0.2,
                        max_tokens=8000,
                        stream=False
                    )
                    self.rate_limiter.reset()
                    modified_code = response.choices[0].message.content.strip()
                    break
                except Exception as e:
                    if "429" in str(e):  # Rate limit error
                        if not self.rate_limiter.handle_error():
                            print("\nRate limit reached. Modification aborted.")
                            self.restore_backup(backup_file)
                            return False
                        continue
                    retries += 1
                    if retries >= max_retries:
                        print(f"\nFailed to generate modifications: {e}")
                        self.restore_backup(backup_file)
                        return False
            
            # Validate the modified code
            print("\nValidating modified code...")
            if not self._validate_modified_code(modified_code):
                print("\nValidation failed. Restoring from backup...")
                self.restore_backup(backup_file)
                return False

            # Save the modified code
            print("\nApplying modifications...")
            with open(__file__, 'w') as f:
                f.write(modified_code)

            # Test the modifications
            print("\nTesting modified code...")
            test_result = await self._test_modified_code()
            
            if not test_result:
                print("\nModified code failed testing. Restoring from backup...")
                self.restore_backup(backup_file)
                return False

            print("\nModification successful! Please restart the assistant to apply changes.")
            return True

        except Exception as e:
            print(f"\nError during self-modification: {e}")
            if os.path.exists(backup_file):
                self.restore_backup(backup_file)
            return False

    def _validate_modified_code(self, code: str) -> bool:
        """Validate the modified code"""
        try:
            # Check for required components
            required_patterns = [
                r'class\s+CommandResult',
                r'class\s+AIComputerAssistant',
                r'def\s+execute_system_command',
                r'def\s+get_response',
                r'def\s+main',
                'InferenceClient',
                'asyncio.run(main())'
            ]
            
            for pattern in required_patterns:
                if not re.search(pattern, code):
                    print(f"\nValidation failed: Missing {pattern}")
                    return False

            # Try to compile the code
            compile(code, '<string>', 'exec')
            return True
        except Exception as e:
            print(f"\nCode validation error: {e}")
            return False

    async def _test_modified_code(self) -> bool:
        """Test the modified code"""
        try:
            # Test basic commands
            test_commands = [
                "echo 'test'",
                "pwd",
                "ls"
            ]

            for cmd in test_commands:
                result = await self.execute_system_command(cmd)
                if not result.was_successful():
                    return False

            return True
        except Exception as e:
            print(f"\nTest failed: {e}")
            return False

async def main():
    assistant = AIComputerAssistant()
    current_mode = None
    
    def handle_mode(mode: str) -> Optional[str]:
        if mode.lower() in ['1', 'chat', 'c']:
            return 'chat'
        elif mode.lower() in ['2', 'system', 's']:
            return 'system'
        elif mode.lower() in ['3', 'modify', 'm']:
            return 'modify'
        elif mode.lower() in ['4', 'exit', 'quit', 'q']:
            return 'exit'
        return None

    print("\nWelcome to AI Computer Assistant!")
    print(f"Running as user: {assistant.user}")
    print(f"Operating System: {assistant.os_type}")
    print("\nYou can type 'switch' to change modes, 'exit' to quit")
    print("Press Ctrl+C at any time to interrupt operation")
    
    while True:
        try:
            while current_mode is None:
                print("\n1. Chat mode (text only)")
                print("2. System control mode")
                print("3. Self-modification mode")
                print("4. Exit")
                choice = input("\nSelect mode (1-4): ").strip().lower()
                current_mode = handle_mode(choice)
                if current_mode == 'exit':
                    print("\nGoodbye!")
                    return
                elif current_mode == 'modify':
                    modification_prompt = input("\nEnter modification request: ").strip()
                    if modification_prompt:
                        if await assistant.modify_self(modification_prompt):
                            return  # Exit to apply changes
                    current_mode = None
                    continue
                elif current_mode is None:
                    print("\nInvalid mode selected. Please try again.")

            prompt = "\nEnter your request ('switch' for mode change, 'exit' to quit): "
            user_input = input(prompt).strip()

            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                return
            elif user_input.lower() == 'switch':
                current_mode = None
                continue
            elif user_input.lower() == 'history':
                print("\nCommand History:")
                for idx, result in enumerate(assistant.command_history[-10:], 1):
                    status = "✓" if result.was_successful() else "✗"
                    print(f"{idx}. [{status}] {result.command}")
                continue
            elif not user_input:
                continue

            if current_mode == 'chat':
                await assistant.get_response(user_input)
            else:  
                await assistant.execute_task(user_input)
            
        except KeyboardInterrupt:
            print("\n\nOperation interrupted.")
            user_choice = input("\nEnter 'continue' to resume, 'skip' to move to next task, or press Enter for new request: ").strip().lower()
            if user_choice == 'continue':
                continue
            elif user_choice == 'skip' and assistant.task_stack:
                assistant.task_stack.pop()
                continue
            elif user_choice == '':
                pass
            else:
                print("\nResuming operation...")
                continue
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")
            continue

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    except Exception as e:
        print(f"\nCritical error: {str(e)}")

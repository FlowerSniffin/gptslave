# GPTSlave: README

## Overview

**GPTSlave** is an intelligent Python-based assistant designed to automate system interactions, analyze command failures, and even modify its own functionality on the fly. With the power of Hugging Face's Inference API, **GPTSlave** acts as your personal AI assistant for executing commands, troubleshooting, and task automation—all while maintaining high security and flexibility.

## Features

- **Rate-Limiting and Error Handling**: Implements rate-limiting with exponential backoff to manage API calls efficiently and handle throttling gracefully.
- **Command Execution**: Executes system commands with real-time output streaming. Includes security measures to validate paths and prevent dangerous operations.
- **Failure Analysis**: Automatically analyzes failed commands and suggests fixes using AI-powered reasoning.
- **Backup and Restore**: Creates backups of its code and restores them if self-modifications fail.
- **Task Automation**: Handles multi-step workflows with intelligent planning and dynamic adjustments.
- **Self-Modification**: Allows users to request changes to its code, ensuring safety through backups, validation, and testing.
- **Command History**: Maintains a history of executed commands for easy reference and debugging.
- **Interactive Modes**:
  - **Chat Mode**: For general queries and interactive discussions.
  - **System Mode**: Automates tasks and executes shell commands.
  - **Self-Modification Mode**: Dynamically modifies its code based on user prompts.
- **Cross-Platform Support**: Compatible with Linux, macOS, and Windows.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/FlowerSniffin/gptslave.git
   cd gptslave
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up the Hugging Face Inference API key:
   - Sign up at [Hugging Face](https://huggingface.co/) and get an API key.
   - Replace the placeholder API key in the `AIComputerAssistant` class with your own.

## Usage

Run the assistant:
```bash
python gptslave.py
```

### Modes

1. **Chat Mode**: For answering questions, debugging, and troubleshooting.
2. **System Mode**: Executes commands and automates tasks securely.
3. **Self-Modification Mode**: Modifies its code dynamically with user prompts.

Switch between modes using the `switch` command. Exit the assistant with the `exit` command.

### Examples

- Execute a system task:
  ```
  Enter your request: List all files in the current directory
  ```
- Analyze a failed command:
  ```
  Enter your request: Fix "permission denied" error when accessing /restricted
  ```
- Modify functionality:
  ```
  Enter modification request: Add logging for all executed commands
  ```

## Security

- **Path Validation**: Ensures that critical system files and directories are protected.
- **Command Filtering**: Detects and blocks potentially harmful commands.
- **Backup System**: Creates backups before applying changes to its code.

## Limitations

- **Rate Limits**: Depends on the Hugging Face API limits.
- **API Key Required**: A valid Hugging Face API key is necessary for full functionality.
- **Testing Needed**: Thorough testing is recommended after applying self-modifications.

## Contributing

Contributions are welcome! Please submit issues, feature requests, or pull requests to improve the project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


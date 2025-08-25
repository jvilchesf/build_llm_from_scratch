# Agent Guidelines for LLM Project

## Build/Test Commands
- **Run main**: `python main.py`
- **Install dependencies**: `uv sync` or `pip install -e .`
- **Single test**: No test framework configured yet
- **Type checking**: No mypy/pyright configured

## Code Style
- **Imports**: Standard library first, third-party second, local imports last with blank lines between groups
- **Formatting**: 4 spaces indentation, functions have docstrings with triple quotes
- **Types**: Use type hints consistently (e.g., `def __init__(self, stride: int = 1)`)
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Line length**: No strict limit observed, keep reasonable
- **Error handling**: Use explicit error handling where needed
- **Comments**: Minimal inline comments, rely on docstrings

## Project Structure
- Main ML training code in `main.py`
- Custom tokenizer and data loader classes in separate modules
- Dependencies: torch, tiktoken for tokenization
- Data downloaded to `source/` directory

## Dependencies
- **Core**: torch>=2.7.1, tiktoken>=0.9.0
- **Python**: >=3.12 required
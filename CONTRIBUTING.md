# Contributing to motcpp

Thank you for your interest in contributing to motcpp! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/motcpp.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes: `cmake --build build && ctest`
6. Commit with clear messages
7. Push to your fork: `git push origin feature/your-feature-name`
8. Open a Pull Request

## Code Style

- Follow the existing code style
- Use `clang-format` for formatting (configuration in `.clang-format`)
- Write clear, self-documenting code
- Add comments for complex logic

## Testing

- Add unit tests for new features
- Ensure all tests pass: `ctest --output-on-failure`
- Maintain or improve code coverage

## Documentation

- Update relevant documentation files
- Add docstrings for new public APIs
- Update examples if needed

## Pull Request Process

1. Ensure your code builds on Linux, macOS, and Windows
2. All tests must pass
3. Documentation must be updated
4. Follow the PR template
5. Be responsive to feedback

## Questions?

Open an issue or discussion on GitHub.

Thank you for contributing!

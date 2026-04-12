# Contributing to MeaningFlow

Thank you for your interest in contributing to MeaningFlow!

## Ways to Contribute

- **Bug Reports**: Open an issue describing the bug, including steps to reproduce
- **Feature Requests**: Open an issue describing the feature and its use case
- **Code Contributions**: Submit a pull request with your changes
- **Documentation**: Help improve docs, examples, or tutorials
- **Use Cases**: Share how you're using MeaningFlow in your work

## Development Setup

```bash
# Clone the repository
git clone https://github.com/Bodhi8/Meaning-Flow.git
cd Meaning-Flow

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode
pip install -e ".[dev]"
```

## Code Style

- We use [Black](https://github.com/psf/black) for code formatting
- We use [Ruff](https://github.com/astral-sh/ruff) for linting
- Type hints are encouraged

```bash
# Format code
black meaningflow/

# Lint
ruff check meaningflow/
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Questions?

Open an issue or start a discussion. We're happy to help!

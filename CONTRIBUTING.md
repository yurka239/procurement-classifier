# Contributing to Procurement Classifier

Thank you for your interest in contributing to Procurement Classifier!

## How to Contribute

### Reporting Issues

1. Check existing issues first to avoid duplicates
2. Use a clear, descriptive title
3. Include:
   - Python version
   - Operating system
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages (if any)

### Suggesting Features

1. Open an issue with `[Feature Request]` prefix
2. Describe the use case
3. Explain how it would benefit users

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Test thoroughly
5. Submit a pull request with a clear description

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add comments for complex logic
- Keep functions focused and small

## Project Structure

```
src/                    # Core modules
├── ai_handler.py       # OpenAI API integration
├── attribute_config.py # Attribute definitions
├── classifier_engine.py # Main classification logic
├── normalizer.py       # Fingerprinting engine
└── ...

app.py                  # Main Streamlit application
```

## Testing

Before submitting:
1. Test with sample data in `_Templates/`
2. Verify attribute extraction works
3. Check classification accuracy
4. Ensure no API keys are exposed

## Questions?

Open an issue with `[Question]` prefix.

---

Thank you for helping improve Procurement Classifier!

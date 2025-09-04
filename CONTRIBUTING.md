# Contributing to Attention-Based SAC Portfolio Allocator

Thank you for your interest in contributing to this project! We welcome contributions from the community and appreciate your help in making this project better.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Git
- Basic understanding of reinforcement learning and portfolio optimization

### Development Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/attention-sac-portfolio.git
   cd attention-sac-portfolio
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## 🛠️ Development Workflow

### Code Style
We use the following tools to maintain code quality:
- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking
- **Pytest** for testing

Run these before committing:
```bash
black src/ examples/ tests/
flake8 src/ examples/ tests/
mypy src/
pytest tests/
```

### Testing
- Write tests for new functionality
- Ensure all existing tests pass
- Aim for >80% code coverage
- Use descriptive test names

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_attention_actor.py
```

### Documentation
- Update docstrings for new functions/classes
- Add type hints to all functions
- Update README.md if adding new features
- Include examples in docstrings

## 📝 Types of Contributions

### 🐛 Bug Reports
When reporting bugs, please include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Error messages/logs

### ✨ Feature Requests
For new features:
- Describe the feature and its benefits
- Provide use cases
- Consider implementation complexity
- Check for existing similar requests

### 🔧 Code Contributions

#### Areas for Contribution
- **New attention mechanisms**: Different attention architectures
- **Additional regime detection**: More sophisticated regime classification
- **Enhanced visualizations**: New plotting capabilities
- **Performance optimizations**: Faster training/inference
- **Documentation**: Better examples and tutorials
- **Tests**: More comprehensive test coverage

#### Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, well-documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   pytest tests/
   black src/ examples/ tests/
   flake8 src/ examples/ tests/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request**
   - Use descriptive title
   - Provide detailed description
   - Link related issues
   - Include screenshots for UI changes

## 📋 Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)

## Related Issues
Closes #issue_number
```

## 🏗️ Project Structure

Understanding the codebase:
```
src/
├── agents/          # SAC agent implementations
├── nets/           # Neural network architectures
├── envs/           # Environment and reward functions
├── utils/          # Utility functions and visualizations
└── eval/           # Evaluation metrics and tools

examples/           # Demo scripts and tutorials
configs/            # Configuration files
tests/              # Unit and integration tests
docs/               # Documentation
```

## 🧪 Testing Guidelines

### Unit Tests
- Test individual functions/classes
- Use descriptive test names
- Test edge cases and error conditions
- Mock external dependencies

### Integration Tests
- Test component interactions
- Test end-to-end workflows
- Use realistic data

### Example Test Structure
```python
def test_attention_actor_forward():
    """Test attention actor forward pass."""
    actor = AttentionActor(obs_dim=50, n_assets=5)
    obs = torch.randn(1, 50)
    
    output = actor(obs)
    
    assert output['weights'].shape == (1, 5)
    assert torch.allclose(output['weights'].sum(dim=-1), torch.ones(1))
    assert 'regime_info' in output
```

## 📚 Documentation Standards

### Docstring Format
```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Brief description of function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When invalid input provided
        
    Example:
        >>> result = function_name(1, 2)
        >>> print(result)
        3
    """
```

### Type Hints
Always include type hints:
```python
from typing import Dict, List, Optional, Tuple

def process_data(
    data: List[float], 
    config: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, float]:
    """Process data with optional configuration."""
    pass
```

## 🎯 Performance Considerations

### Code Performance
- Use vectorized operations (NumPy/PyTorch)
- Avoid unnecessary loops
- Profile code for bottlenecks
- Consider memory usage

### Model Performance
- Monitor training metrics
- Use appropriate batch sizes
- Consider model complexity vs performance
- Document performance characteristics

## 🔍 Code Review Process

### For Contributors
- Respond to review feedback promptly
- Make requested changes
- Ask questions if feedback is unclear
- Be open to suggestions

### For Reviewers
- Be constructive and respectful
- Focus on code quality and correctness
- Check for security issues
- Ensure tests are adequate

## 🐛 Issue Guidelines

### Bug Reports
Use the bug report template:
```markdown
**Describe the bug**
Clear description of the issue

**To Reproduce**
Steps to reproduce the behavior

**Expected behavior**
What you expected to happen

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.8.5]
- Package versions: [e.g., torch 1.9.0]

**Additional context**
Any other relevant information
```

### Feature Requests
Use the feature request template:
```markdown
**Is your feature request related to a problem?**
Description of the problem

**Describe the solution you'd like**
Clear description of the desired solution

**Describe alternatives you've considered**
Other solutions you've thought about

**Additional context**
Any other context about the feature request
```

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Email**: For security issues or private matters

## 🏆 Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Thank You

Thank you for contributing to this project! Your efforts help make this tool better for everyone in the community.

---

**Happy coding! 🚀**

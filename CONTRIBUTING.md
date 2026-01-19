# Contributing to NNMM.jl

Thank you for your interest in contributing to NNMM.jl! This document provides guidelines for contributing to the project.

## Reporting Bugs

If you find a bug, please open an issue on GitHub with:

1. **Title**: A clear, descriptive title
2. **Description**: What you expected to happen vs. what actually happened
3. **Reproduction steps**: Minimal code to reproduce the issue
4. **Environment**: Julia version, OS, and NNMM.jl version
5. **Error message**: Full error output if applicable

Example:
```julia
using NNMM
# Minimal code that triggers the bug
```

## Suggesting Features

Feature requests are welcome! Please open an issue with:

1. **Use case**: What problem are you trying to solve?
2. **Proposed solution**: How do you envision the feature working?
3. **Alternatives**: Any alternative solutions you've considered

## Pull Requests

### Before You Start

1. Check existing issues and PRs to avoid duplicate work
2. For large changes, open an issue first to discuss the approach
3. Fork the repository and create a feature branch

### Development Setup

```julia
using Pkg
Pkg.develop(path="/path/to/your/fork/NNMM.jl")
```

### Code Style

- Follow Julia's [style guide](https://docs.julialang.org/en/v1/manual/style-guide/)
- Use descriptive variable and function names
- Add docstrings to exported functions
- Keep lines under 100 characters when practical

### Testing

All changes must pass existing tests:

```julia
using Pkg
Pkg.test("NNMM")
```

For new features, add corresponding tests in the `test/` directory.

### Submitting

1. Ensure all tests pass locally
2. Update documentation if needed
3. Add an entry to `CHANGELOG.md` under `[Unreleased]`
4. Create a pull request with a clear description of changes

## Questions?

- Open a [GitHub issue](https://github.com/reworkhow/NNMM.jl/issues)
- Contact: <qtlcheng@ucdavis.edu>

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

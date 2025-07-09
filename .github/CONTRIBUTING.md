# Contributing to AI From Scratch to Scale

Thank you for your interest in contributing to this educational project! This guide outlines our development workflow, standards, and expectations for contributors.

## Project Philosophy

### Core Roles
- **You (The Contributor)** are the **Project Architect**: Responsible for high-level design, making key decisions, asking critical questions, and ensuring the final code aligns with our learning objectives. You are the ultimate quality gate.
- **AI Assistant (GitHub Copilot)** is the **Lead Engineer / Pair Programmer**: Assists with code generation, pattern implementation, and development acceleration. Requires your direct oversight and validation.

### Educational Focus
Our code is not just an implementation—it's a primary learning artifact. We prioritize:
- **Clarity over cleverness**: Educational value over performance optimization
- **Mathematical rigor**: Detailed derivations and intuitive explanations
- **Historical context**: Understanding *why* models were invented, not just *how* they work
- **Practical wisdom**: Real datasets, common pitfalls, and practical considerations

## Development Workflow

### Git Workflow & Branching Strategy

Each model implementation follows a feature branch workflow:

1. **Create Feature Branch**: `git checkout -b feature/XX_ModelName`
   - Branch naming: `feature/01_Perceptron`, `feature/05_LeNet-5`, etc.
   - Always branch from `main`

2. **Development Process**: Iterative development with frequent commits

3. **Pull Request**: Create PR to merge feature branch into `main`

4. **Cleanup**: Delete feature branch after successful merge

### Commit Message Convention

We follow the **Conventional Commits** specification:

**Format**: `type: description`

**Common Types**:
- `feat`: A new feature (e.g., `feat: add model.py for Perceptron`)
- `fix`: A bug fix (e.g., `fix: correct loss calculation in train.py`)
- `docs`: Documentation-only changes (e.g., `docs: update README with setup instructions`)
- `style`: Formatting changes that don't affect code meaning
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `test`: Adding missing tests or correcting existing ones
- `chore`: Changes to build process or auxiliary tools

## Model Development Process

### Step 1: Setup and Scaffolding

1. **Create the Branch**: `git checkout -b feature/XX_ModelName`

2. **Create Virtual Environment**:
   ```bash
   cd XX_ModelName
   python -m venv .venv
   # Activate based on your system (see activation scripts)
   ```

3. **Create Directory Structure**:
   ```
   XX_ModelName/
   ├── src/
   │   ├── __init__.py
   │   ├── config.py          # Configuration and hyperparameters
   │   ├── data_loader.py     # Data loading and preprocessing
   │   ├── model.py           # Model implementation
   │   ├── train.py           # Training logic
   │   ├── evaluate.py        # Evaluation and testing
   │   └── visualize.py       # Plotting and visualization
   ├── data/                  # Dataset storage
   ├── notebooks/             # Jupyter notebooks for exploration
   ├── outputs/               # Generated outputs (models, plots, logs)
   │   ├── logs/
   │   ├── models/
   │   └── visualizations/
   ├── requirements.txt       # Dependencies
   └── README.md             # Model-specific documentation
   ```

4. **Initial Commit**: `git commit -m "chore: initialize project structure for ModelName"`

### Step 2: Iterative Development (Copilot-Optimized)

**Development Order** (recommended):
1. `src/config.py` - Configuration and hyperparameters
2. `src/data_loader.py` - Data handling
3. `src/model.py` - Core model implementation
4. `src/train.py` - Training orchestration
5. `src/evaluate.py` - Evaluation and testing
6. `src/visualize.py` - Plotting and analysis

**Development Process**:
1. **Start with scaffolding**: Create file structure with docstrings and type hints
2. **Implement incrementally**: Use Copilot for method bodies, error handling, logging
3. **Leverage context**: Copilot understands project patterns and suggests consistent code
4. **Commit frequently**: Small, logical commits as features are completed

**Review Checklist** for each component:
- **Correctness**: Does it meet requirements? Any logical flaws?
- **Clarity**: Does it follow our "Code as a Learning Tool" philosophy?
- **Compliance**: Does it follow our coding standards (naming, docstrings, type hints)?
- **Educational Value**: Is it easy to understand and learn from?

### Step 3: Integration & Testing

1. **Execute the Code**: Run `python src/train.py`
2. **Debug Collaboratively**: Use error tracebacks and Copilot assistance for debugging
3. **Analyze Outputs**: Review console output, `training.log`, and saved visualizations
4. **Validate Learning Objectives**: Ensure implementation achieves educational goals

### Step 4: Documentation & Finalization

1. **Update Dependencies**: `pip freeze > requirements.txt`
2. **Create Comprehensive README**: Follow the template below
3. **Create Pull Request**: Include link to learning objectives
4. **Final Review**: Ensure all quality gates are met

## README Template for Each Model

Each model's `README.md` must include:

### Required Sections:

```markdown
# Module X, Model Y: [Model Name]

## Introduction
Brief summary of the model, historical context, and significance.

## Core Innovation
Clear explanation of the key new idea introduced by this model.

## Mathematical Foundation
- Key equations and derivations
- Intuitive explanations of mathematical concepts
- Visual diagrams where helpful

## How to Run This Code

### Prerequisites
- Python 3.8+
- Any non-Python dependencies

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
python src/train.py
```

### Evaluation
```bash
python src/evaluate.py
```

## Results and Analysis

### The "Success" Case
Description of experiment where the model excels:
- Final metrics (accuracy, loss)
- Key visualizations with analysis
- Explanation of *why* it succeeded

### The "Failure" Case
Description of experiment showing model limitations:
- Metrics demonstrating failure
- Visualizations diagnosing the problem
- Analysis of fundamental limitations

## Key Takeaways
- Bulleted list of most important lessons learned
- Mathematical insights
- Historical significance
- Connections to future models

## Project Structure
Brief explanation of file organization and key components.

## License
This project is licensed under the MIT License.
```

## Workflow for "Conceptual" Models

For models marked as "Conceptual":

1. Follow Step 1 (setup and scaffolding)
2. Focus primarily on comprehensive README.md file
3. Include any supporting diagrams or simplified code examples
4. Commit documentation and supporting files
5. Follow standard merge process

## Code Quality Standards

All contributions must meet these standards:

### Type Safety & Documentation
- Comprehensive type hints for all functions and classes
- Google-style docstrings with mathematical explanations
- Clear parameter and return value descriptions

### Testing & Validation
- Unit tests for core model functions
- Integration tests for complete workflows
- Edge case testing and error handling

### Performance & Optimization
- **Early Models (NumPy-based)**: Focus on algorithmic clarity
- **Later Models (Framework-based)**: Apply modern optimization practices
- Profile code for bottlenecks when necessary

### Educational Excellence
- Prioritize readability and educational value
- Include extensive comments explaining mathematical operations
- Add assertions to validate intermediate computations
- Create modular, testable components

## Getting Help

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Code Review**: All PRs require review before merging

## Recognition

Contributors will be recognized in:
- Project README acknowledgments
- Individual model documentation
- Release notes and project updates

---

*By contributing to this project, you help create a valuable educational resource for the global ML community. Thank you for your efforts in advancing AI education!*

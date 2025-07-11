"""
AI From Scratch Shared Utilities Setup

This package provides common utilities for the AI-From-Scratch-to-Scale project,
including standardized W&B integration patterns and visualization helpers.

Installation:
    pip install -e ai_from_scratch_shared/

Educational Focus:
    This package demonstrates professional Python packaging practices
    and provides reusable components for ML experiment tracking.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="ai_from_scratch_shared",
    version="1.0.0",
    description="Shared utilities for AI-From-Scratch-to-Scale educational project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AI-From-Scratch-to-Scale Project",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "wandb>=0.16.0",
        "matplotlib>=3.5.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
    ],
    keywords="machine-learning, education, experiment-tracking, wandb, neural-networks",
    project_urls={
        "Source": "https://github.com/jfransella/AI-From-Scratch-to-Scale",
        "Documentation": "https://github.com/jfransella/AI-From-Scratch-to-Scale/blob/main/ai_from_scratch_shared/README.md",
    },
)

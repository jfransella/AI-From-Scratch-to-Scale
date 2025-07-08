#!/bin/bash
# Cross-platform virtual environment activation script
echo "Activating Perceptron virtual environment..."

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source .venv/Scripts/activate
else
    # macOS/Linux
    source .venv/bin/activate
fi

echo "Virtual environment activated!"

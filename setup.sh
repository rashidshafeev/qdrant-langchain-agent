#!/bin/bash

# Display colorful banner
echo -e "\033[1;36m"
echo "==============================================="
echo "  Setting up Qdrant LangChain Agent Environment"
echo "==============================================="
echo -e "\033[0m"

# Check Python version
echo -e "\033[1;33m[1/5] Checking Python version...\033[0m"
python_version=$(python3 --version 2>&1)

if [[ $? -ne 0 || ! $python_version =~ "Python 3" ]]; then
    echo -e "\033[1;31mError: Python 3 is required but not found.\033[0m"
    echo "Please install Python 3.8 or newer and try again."
    exit 1
fi
echo -e "\033[1;32mFound $python_version\033[0m"

# Create virtual environment
echo -e "\033[1;33m[2/5] Creating virtual environment...\033[0m"
if [ -d "venv" ]; then
    echo "Virtual environment already exists."
    echo -e "Do you want to recreate it? \033[1;33m[y/N]\033[0m"
    read -r recreate
    if [[ $recreate =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf venv
        python3 -m venv venv
        if [ $? -ne 0 ]; then
            echo -e "\033[1;31mError creating virtual environment.\033[0m"
            exit 1
        fi
    fi
else
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "\033[1;31mError creating virtual environment.\033[0m"
        exit 1
    fi
fi
echo -e "\033[1;32mVirtual environment ready\033[0m"

# Activate virtual environment
echo -e "\033[1;33m[3/5] Activating virtual environment...\033[0m"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Linux/Mac
    source venv/bin/activate
fi
echo -e "\033[1;32mVirtual environment activated\033[0m"

# Install dependencies
echo -e "\033[1;33m[4/5] Installing dependencies...\033[0m"
pip install -U pip wheel setuptools
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "\033[1;31mError installing dependencies.\033[0m"
    exit 1
fi
echo -e "\033[1;32mDependencies installed successfully\033[0m"

# Set up .env file
echo -e "\033[1;33m[5/5] Setting up environment variables...\033[0m"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "\033[1;32mCreated .env file from example\033[0m"
    echo -e "\033[1;33mPlease edit .env file to configure your API keys and settings\033[0m"
else
    echo -e "\033[1;32m.env file already exists\033[0m"
fi

echo -e "\033[1;36m"
echo "==============================================="
echo "  Setup Complete!"
echo "==============================================="
echo -e "\033[0m"

echo -e "To activate the environment, run:"
echo -e "\033[1;33msource venv/bin/activate\033[0m  # On Linux/Mac"
echo -e "\033[1;33mvenv\\Scripts\\activate\033[0m    # On Windows"
echo ""
echo -e "To use the agent, run:"
echo -e "\033[1;33mpython -m qdrant_agent --help\033[0m"
echo ""
echo -e "Don't forget to edit the \033[1;33m.env\033[0m file to configure your API keys!"

"""Setup script for the qdrant-agent package."""

from setuptools import setup, find_packages

# Read the contents of the README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open("requirements.txt", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="qdrant-agent",
    version="0.1.0",
    description="A LangChain-based agent for interacting with Qdrant vector database",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/rashidshafeev/qdrant-langchain-agent",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "qdrant-agent=qdrant_agent.cli:cli",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
)

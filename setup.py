#!/usr/bin/env python3
"""
Setup script for AI Art Generator
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="ai-art-generator",
    version="0.1.0",
    description="A Python CLI tool for generating visual assets using multiple AI image generation services",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/ai-art-generator",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "ai-art-generator=art-generator:main",
        ],
    },
    python_requires=">=3.12",
    install_requires=[
        "google-genai>=1.20.0",
        "google-cloud-aiplatform>=1.73.0", 
        "openai>=1.88.0",
        "pillow>=11.2.1",
        "python-dotenv>=1.1.0",
        "pyyaml>=6.0.2",
        "requests>=2.32.4",
        "tqdm>=4.67.1",
    ],
    extras_require={
        "dev": [
            "black>=25.1.0",
            "pytest>=8.4.1", 
            "ruff>=0.12.0",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="ai art generation image dalle imagen stable-diffusion",
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.md"],
    },
)
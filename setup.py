"""Setup script for LLM API Server package."""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="llm-api-server",
    version="0.1.0",
    author="LLM API Server Contributors",
    description="Reusable Flask server for LLM backends (Ollama, LM Studio) with tool calling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llm-api-server",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "flask>=3.0.0",
        "flask-cors>=4.0.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "click>=8.1.0",
        "langchain-core>=0.1.0",
    ],
    extras_require={
        "webui": ["open-webui"],
        "dev": ["pytest", "black", "flake8"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="llm openai api flask ollama lmstudio tools",
)

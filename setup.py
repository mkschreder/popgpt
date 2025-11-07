"""Setup script for popgpt package."""

from setuptools import setup, find_packages

with open("requirements.txt", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="popgpt",
    version="2.0.0",
    description="A refactored, production-ready implementation of PopGPT",
    author="PopGPT Contributors",
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "popgpt=popgpt.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

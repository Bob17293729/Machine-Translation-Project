"""Setup script for Machine Translation Project."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="machine-translation",
    version="1.0.0",
    author="Machine Translation Team",
    author_email="",
    description="A modern machine translation system for Chinese-English translation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Machine-Translation-Project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "mt-train=scripts.train:main",
            "mt-eval=scripts.evaluate:main",
            "mt-infer=scripts.inference:main",
        ],
    },
)


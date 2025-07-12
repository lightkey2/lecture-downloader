#!/usr/bin/env python3
"""
Setup script for lecture_downloader package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lecture_downloader",
    version="1.0.11",
    author="dan-dev-ml",
    author_email="dan.dev.ml@gmail.com",
    url="https://github.com/dan-dev-ml/lecture-downloader",
    description="A comprehensive toolkit for downloading, merging, and transcribing lecture videos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "click>=8.0.0",
        "pandas>=1.3.0",
        "python-dotenv>=0.19.0",
        "aiofiles>=0.8.0",
        "aiohttp>=3.8.0",
        "faster-whisper>=0.10.0",
        # "google-cloud-speech>=2.16.0",
        # "google-cloud-storage>=2.7.0",
        "IPython>=7.0.0",
        "ipywidgets>=7.6.0",
        "jupyter",
        "notebook"
    ],
    extras_require={
        "gcloud": [
            "google-cloud-speech>=2.16.0",
            "google-cloud-storage>=2.7.0",
        ],
        "jupyter": [
            "IPython>=7.0.0",
            "ipywidgets>=7.6.0",
            "jupyter",
            "notebook",
        ],
        "all": [
            "google-cloud-speech>=2.16.0",
            "google-cloud-storage>=2.7.0",
            "IPython>=7.0.0",
            "ipywidgets>=7.6.0",
            "jupyter",
            "notebook",
        ],
    },
    entry_points={
        "console_scripts": [
            "lecture-downloader=lecture_downloader.cli:cli",
        ],
    },
)

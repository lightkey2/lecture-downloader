#!/usr/bin/env python3
"""
Setup script for lecture_downloader package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lecture_downloader",
    version="1.1.4",
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
        "click>=7.0",
        "pandas>=1.0",
        "python-dotenv>=0.15",
        "aiofiles>=0.6",
        "aiohttp>=3.6",
        "faster-whisper==1.1.1",
        "imageio-ffmpeg>=0.4.0",
        "rich>=10.0"
        # "google-cloud-speech>=2.16.0",
        # "google-cloud-storage>=2.7.0",
    ],
    extras_require={
        "gcloud": [
            "google-cloud-speech>=2.16.0",
            "google-cloud-storage>=2.7.0",
        ],
        "jupyter": [
            "IPython==8.37.0",
            "ipywidgets==8.1.7",
            "jupyter==1.1.1",
            "notebook==7.4.3",
        ],
        "all": [
            "google-cloud-speech>=2.16.0",
            "google-cloud-storage>=2.7.0",
            "IPython==8.37.0",
            "ipywidgets==8.1.7",
            "jupyter==1.1.1",
            "notebook==7.4.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "lecture-downloader=lecture_downloader.cli:cli",
        ],
    },
)

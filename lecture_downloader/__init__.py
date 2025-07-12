#!/usr/bin/env python3
"""
Lecture Downloader - A comprehensive toolkit for downloading, merging, and transcribing lecture videos.

This package provides both class-based and functional APIs for processing lecture videos:

Class-based API (Primary):
    from lecture_downloader import LectureProcessor
    processor = LectureProcessor()
    results = processor.download_lectures(links, titles)

Functional API (Convenience):
    import lecture_downloader as ld
    results = ld.download_lectures(links, titles)

Features:
- Download lecture videos from Canvas
- Merge videos by module with chapter markers
- Transcribe videos using Google Cloud Speech-to-Text or faster-whisper
- Automatic transcription method detection based on environment
- Flexible input handling (files, strings, lists, dicts)
- Async processing with hidden complexity
"""

# Class-based API (primary)
from .processor import LectureProcessor
from .merger import merge_videos
from .pipeline import process_pipeline
# Functional API (convenience)
from .downloader import download_lectures
from .transcriber import transcribe_videos

# Version info
__version__ = "1.0.11"
__author__ = "dan-dev-ml"
__email__ = "dan.dev.ml@gmail.com"

# Public API exports
__all__ = [
    # Class API
    "LectureProcessor",
    # Functional API
    "download_lectures",
    "merge_videos", 
    "transcribe_videos",
    "process_pipeline",
    # Version info
    "__version__",
]

# Package metadata
__title__ = "lecture_downloader"
__description__ = "A toolkit for downloading, and transcribing videos from lecture platforms like Canvas and Brightspace"
__url__ = "https://github.com/dan-dev-ml/lecture-downloader"
__license__ = "MIT"

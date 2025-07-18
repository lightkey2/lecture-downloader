#!/usr/bin/env python3
"""
Pipeline functionality for lecture_downloader package.
Orchestrates the complete workflow: download -> merge -> transcribe.
"""

import os
import asyncio
from typing import Dict, List, Optional, Union

from .merger import _merge_all_modules_async, _print_merge_mapping
from .transcriber import _transcribe_videos_async, _detect_input_path as transcriber_detect_input_path
from .utils import create_directory_structure, check_dependencies, detect_transcription_method
from .downloader import _parse_links_input, _parse_titles_input, _map_units_to_modules, _generate_lecture_data, _download_lectures_async, _print_download_mapping


async def _process_pipeline_async(
    links: Union[str, List[str]],
    titles: Union[str, List[str], Dict[str, List[str]], None] = None,
    output_dir: str = "lecture_processing",
    max_download_workers: int = 5,
    max_transcribe_workers: int = 3,
    transcription_method: str = "auto",
    language: str = "en-US",
    inject_subtitles: bool = True,
    download_only: bool = False,
    merge_only: bool = False,
    transcribe_only: bool = False,
    verbose: bool = False,
    model_size_or_path: str = "base"
) -> Dict[str, Dict[str, List[str]]]:
    """Complete pipeline with async orchestration."""
    
    results = {
        "download": {"successful": [], "failed": []},
        "merge": {"successful": [], "failed": []},
        "transcribe": {"successful": [], "failed": []}
    }
    
    if verbose:
        print("Starting lecture processing pipeline...")
        print(f"Download only: {download_only}")
        print(f"Merge only: {merge_only}")
        print(f"Transcribe only: {transcribe_only}")
    
    # Create unified directory structure
    dir_structure = create_directory_structure(output_dir)
    actual_download_dir = dir_structure['downloads']
    actual_merged_dir = dir_structure['merged']
    actual_transcripts_dir = dir_structure['transcripts']
    
    if verbose:
        print(f"Created unified directory structure in: {output_dir}")
        print(f"   ├── 01_downloads/     (individual lecture files)")
        print(f"   ├── 02_merged/        (merged module videos)")
        print(f"   └── 03_transcripts/   (SRT and TXT files)")
    
    # Step 1: Download lectures (unless merge_only or transcribe_only)
    if not merge_only and not transcribe_only:
        print("Step 1: Downloading lectures...")
        
        # Parse and normalize inputs
        parsed_links = _parse_links_input(links, verbose)
        parsed_titles = _parse_titles_input(titles, verbose)
        
        # Map units to modules if titles provided
        titles_mapping = None
        if parsed_titles:
            titles_mapping = _map_units_to_modules(parsed_links, parsed_titles, verbose)
        
        # Generate lecture data
        lectures = _generate_lecture_data(parsed_links, titles_mapping, verbose)
        
        # Show download mapping
        _print_download_mapping(lectures, actual_download_dir, verbose)
        
        # Execute download
        results["download"] = await _download_lectures_async(
            lectures, actual_download_dir, max_download_workers, use_custom_titles=True, verbose=verbose
        )
        
        if download_only:
            return results
    
    # Step 2: Merge videos (unless download_only or transcribe_only)
    if not download_only and not transcribe_only:
        print("\nStep 2: Merging videos by module...")
        
        # Use the download directory as input for merging
        input_for_merge = actual_download_dir if not merge_only else actual_download_dir
        
        # Check if this is a single file scenario (no merging needed)
        direct_mp4_files = []
        module_directories = []
        
        for item in os.listdir(input_for_merge):
            item_path = os.path.join(input_for_merge, item)
            if os.path.isdir(item_path):
                has_mp4 = any(f.lower().endswith('.mp4') for f in os.listdir(item_path))
                if has_mp4:
                    module_directories.append((item, item_path))
            elif item.lower().endswith('.mp4'):
                direct_mp4_files.append(item)
        
        # Determine if we should skip merging
        is_single_file_scenario = len(direct_mp4_files) > 0 and len(module_directories) == 0
        
        if is_single_file_scenario:
            if verbose:
                print(f"Detected single file scenario: {len(direct_mp4_files)} file(s) directly in downloads")
                print("Skipping merge step - no module directories found")
            modules_to_process = []
        else:
            modules_to_process = module_directories
        
        # Show merge mapping
        _print_merge_mapping(modules_to_process, input_for_merge, actual_merged_dir, verbose)
        
        if is_single_file_scenario:
            # Set empty results for merge step since no merging is needed
            results["merge"] = {"successful": [], "failed": []}
        else:
            results["merge"] = await _merge_all_modules_async(input_for_merge, actual_merged_dir, verbose)
        
        if merge_only:
            return results
    
    # Step 3: Transcribe videos (unless download_only or merge_only)
    if not download_only and not merge_only:
        print("\nStep 3: Transcribing videos...")
        
        # Auto-detect transcription method if needed
        if transcription_method == "auto":
            transcription_method = detect_transcription_method()
            if verbose:
                print(f"Auto-detected transcription method: {transcription_method}")
        
        # Check if transcription is possible
        if transcription_method == "gcloud":
            bucket_name = os.environ.get('GOOGLE_CLOUD_STORAGE_BUCKET')
            if not bucket_name:
                print("Warning: Google Cloud Storage bucket not configured, skipping transcription")
                return results
        
        # Smart input selection for transcription
        if not transcribe_only:
            # Check if merged directory has any MP4 files
            merged_has_mp4 = False
            if os.path.exists(actual_merged_dir) and os.path.isdir(actual_merged_dir):
                merged_has_mp4 = any(f.lower().endswith('.mp4') for f in os.listdir(actual_merged_dir))
            
            if merged_has_mp4:
                # Use merged directory if it has videos
                input_for_transcribe = actual_merged_dir
                if verbose:
                    print("Using merged videos for transcription")
            else:
                # Fall back to download directory for single file scenarios
                input_for_transcribe = actual_download_dir
                if verbose:
                    print("No merged videos found, using download directory for transcription")
        else:
            # transcribe_only mode - use merged directory as specified
            input_for_transcribe = actual_merged_dir
        
        # Collect video files for display
        videos_to_transcribe = []
        if os.path.isfile(input_for_transcribe) and input_for_transcribe.lower().endswith('.mp4'):
            videos_to_transcribe.append(input_for_transcribe)
        elif os.path.isdir(input_for_transcribe):
            for file in os.listdir(input_for_transcribe):
                if file.lower().endswith('.mp4'):
                    videos_to_transcribe.append(os.path.join(input_for_transcribe, file))
        
        # Sort videos for consistent display
        import re
        def extract_module_number_from_filename(filename: str) -> int:
            try:
                match = re.search(r'Module\s+(\d+)', filename, re.IGNORECASE)
                if match:
                    return int(match.group(1))
                return 999
            except (ValueError, AttributeError):
                return 999
        
        videos_to_transcribe.sort(key=lambda x: extract_module_number_from_filename(os.path.basename(x)))
        
        # Show transcription mapping
        # _print_transcribe_mapping(videos_to_transcribe, input_for_transcribe, actual_transcripts_dir, transcription_method, verbose)
        
        results["transcribe"] = await _transcribe_videos_async(
            input_for_transcribe, actual_transcripts_dir, language, 
            transcription_method, max_transcribe_workers, inject_subtitles, verbose, model_size_or_path
        )
    
    return results


# Public functional API
def process_pipeline(
    links: Union[str, List[str]],
    titles: Union[str, List[str], Dict[str, List[str]], None] = None,
    output_dir: str = "downloaded_lectures",
    max_download_workers: int = 5,
    max_transcribe_workers: int = 3,
    transcription_method: str = "auto",
    language: str = "en",
    inject_subtitles: bool = True,
    download_only: bool = False,
    merge_only: bool = False,
    transcribe_only: bool = False,
    verbose: bool = False,
    model_size_or_path: str = "base"
) -> Dict[str, Dict[str, List[str]]]:
    """
    Complete pipeline: download -> merge -> transcribe.
    
    Creates unified directory structure:
    output_dir/
    ├── 01_downloads/
    ├── 02_merged/
    └── 03_transcripts/
    
    Args:
        links: File path to links file, single URL, or list of URLs
        titles: File path to JSON titles file, list of titles, dict mapping, or None
        output_dir: Base output directory
        max_download_workers: Concurrent downloads
        max_transcribe_workers: Concurrent transcriptions
        transcription_method: "auto", "gcloud", or "whisper"
        language: Language code for transcription
        inject_subtitles: Inject SRT into videos
        download_only: Stop after downloading
        merge_only: Only merge (skip download/transcribe)
        transcribe_only: Only transcribe (skip download/merge)
        verbose: Enable progress output
        model_size_or_path: Whisper model size or path to custom model
    
    Returns:
        {
            "download": {"successful": [...], "failed": [...]},
            "merge": {"successful": [...], "failed": [...]},
            "transcribe": {"successful": [...], "failed": [...]}
        }
    """
    # Check dependencies
    check_dependencies()
    
    # Execute pipeline (handles async internally)
    return asyncio.run(_process_pipeline_async(
        links, titles, output_dir, max_download_workers, max_transcribe_workers,
        transcription_method, language, inject_subtitles,
        download_only, merge_only, transcribe_only, verbose, model_size_or_path
    ))

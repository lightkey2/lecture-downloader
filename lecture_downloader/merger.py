#!/usr/bin/env python3
"""
Video merging functionality for lecture_downloader package.
"""

import os
import json
import asyncio
import tempfile
import subprocess
from typing import List, Dict

from .utils import natural_sort_key, format_module_name_with_padding, get_video_duration, get_ffmpeg_exe


def _print_merge_mapping(modules_to_process: List[tuple], input_dir: str, output_dir: str, verbose: bool = False):
    """Print a clean tree view of the merge mapping."""
    # Always show the clean tree view (removed logging dependency)
    
    print(f"Merge Plan:")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Modules: {len(modules_to_process)}")
    print()
    
    for i, (module_name, module_path) in enumerate(modules_to_process):
        is_last_module = i == len(modules_to_process) - 1
        module_prefix = "└── " if is_last_module else "├── "
        
        # Count videos in module
        video_count = 0
        if os.path.exists(module_path) and os.path.isdir(module_path):
            video_count = sum(1 for f in os.listdir(module_path) if f.lower().endswith('.mp4'))
        
        if verbose: print(f"{module_prefix}{module_name} ({video_count} videos)")
        
        # Show video files if not too many
        if video_count <= 5 and os.path.exists(module_path):
            video_files = [f for f in os.listdir(module_path) if f.lower().endswith('.mp4')]
            video_files.sort(key=lambda x: natural_sort_key(x))
            
            for j, video_file in enumerate(video_files):
                is_last_video = j == len(video_files) - 1
                if is_last_module:
                    video_prefix = "    └── " if is_last_video else "    ├── "
                else:
                    video_prefix = "│   └── " if is_last_video else "│   ├── "
                # Remove .mp4 extension for cleaner display
                display_name = os.path.splitext(video_file)[0]
        
        if verbose: 
            if 'video_prefix' in locals():
                # Print video file with prefix
                if is_last_module:
                    print(f"{video_prefix}{display_name}")
                else:
                    print(f"{video_prefix}{display_name}")

        elif video_count > 5:
            # Just show count for many videos
            if is_last_module:
                if verbose: print(f"    └── ... {video_count} video files")
            else:
                if verbose: print(f"│   └── ... {video_count} video files")
    print()


async def _create_concat_file(video_files: List[str], temp_dir: str) -> str:
    """Create a temporary concat file for FFmpeg."""
    concat_file = os.path.join(temp_dir, 'concat_list.txt')
    with open(concat_file, 'w') as f:
        for video_file in video_files:
            # Use absolute path and escape properly for FFmpeg
            abs_path = os.path.abspath(video_file)
            escaped_path = abs_path.replace("'", r"\'").replace("\\", "/")
            f.write(f"file '{escaped_path}'\n")
    return concat_file


async def _create_chapters_file(video_files: List[str], temp_dir: str, verbose: bool = False) -> str:
    """Create a chapters metadata file."""
    chapters_file = os.path.join(temp_dir, 'chapters.txt')
    
    current_time = 0.0
    
    with open(chapters_file, 'w') as f:
        f.write(";FFMETADATA1\n")
        
        for i, video_file in enumerate(video_files):
            duration = get_video_duration(video_file)
            
            filename = os.path.basename(video_file)
            chapter_title = os.path.splitext(filename)[0]
            
            start_time_ms = int(current_time * 1000)
            end_time_ms = int((current_time + duration) * 1000)
            
            f.write(f"\n[CHAPTER]\n")
            f.write(f"TIMEBASE=1/1000\n")
            f.write(f"START={start_time_ms}\n")
            f.write(f"END={end_time_ms}\n")
            f.write(f"title={chapter_title}\n")
            
            current_time += duration
            if verbose:
                print(f"  Chapter {i+1}: {chapter_title} ({current_time:.2f}s)")
    
    return chapters_file


async def _merge_single_module(module_dir: str, output_dir: str, verbose: bool = False) -> bool:
    """Merge all videos in a module directory into a single video with chapters."""
    module_name = os.path.basename(module_dir)
    if verbose: print(f"Processing module: {module_name}")
    
    video_files = []
    for file in os.listdir(module_dir):
        if file.lower().endswith('.mp4'):
            video_files.append(os.path.join(module_dir, file))
    
    if not video_files:
        print(f"  No MP4 files found in {module_dir}")
        return False
    
    video_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    
    if verbose: print(f"  Found {len(video_files)} videos")
    if verbose:
        for i, video in enumerate(video_files, 1):
            print(f"    {i}. {os.path.basename(video)}")
    
    # Create zero-padded module name for better file ordering
    safe_module_name = format_module_name_with_padding(module_name)
    output_file = os.path.join(output_dir, f"{safe_module_name}.mp4")

    with tempfile.TemporaryDirectory() as temp_dir:
        concat_file = await _create_concat_file(video_files, temp_dir)
        chapters_file = await _create_chapters_file(video_files, temp_dir, verbose)
        
        if verbose: print(f"  Merging videos into: {output_file}")
        
        cmd = [
            get_ffmpeg_exe(),
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-i', chapters_file,
            '-map_metadata', '1',
            '-c', 'copy',
            '-y',
            output_file
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                stderr_text = stderr.decode()
                print(f"  ❌ Error merging videos for {module_name}:")
                print(f"     FFmpeg error (exit code {process.returncode})")
                if verbose:
                    print(f"     Error details: {stderr_text.strip()}")
                return False
            
            print(f"  ✅ Successfully created: {output_file}")
            return True
            
        except Exception as e:
            print(f"  ❌ Exception merging videos for {module_name}: {e}")
            return False


async def _merge_all_modules_async(base_dir: str, output_dir: str, verbose: bool = False) -> Dict[str, List[str]]:
    """Merge all module directories in base_dir."""
    os.makedirs(output_dir, exist_ok=True)
    
    results = {"successful": [], "failed": []}
    
    # Find all directories with MP4 files
    modules_to_process = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        
        if not os.path.isdir(item_path):
            continue
        
        has_mp4 = any(f.lower().endswith('.mp4') for f in os.listdir(item_path))
        if not has_mp4:
            if verbose:
                print(f"Skipping {item} (no MP4 files found)")
            continue
        
        modules_to_process.append((item, item_path))
    
    # Process modules sequentially (video merging is CPU intensive)
    for module_name, module_path in modules_to_process:
        success = await _merge_single_module(module_path, output_dir, verbose)
        if success:
            results["successful"].append(module_name)
        else:
            results["failed"].append(module_name)
    
    # Summary
    print(f"Merge Summary: Successful: {len(results['successful'])}, Failed: {len(results['failed'])}")
    
    return results


def _detect_input_directory(base_dir: str, verbose: bool = False) -> str:
    """
    Smart input directory detection for merge operation.
    
    Args:
        base_dir: Base directory to search in
        verbose: Enable verbose output
        
    Returns:
        Path to directory containing videos to merge
    """
    # Check for lecture-downloads subdirectory first
    lecture_downloads_path = os.path.join(base_dir, "lecture-downloads")
    if os.path.exists(lecture_downloads_path) and os.path.isdir(lecture_downloads_path):
        # Check if it contains any directories with MP4 files
        for item in os.listdir(lecture_downloads_path):
            item_path = os.path.join(lecture_downloads_path, item)
            if os.path.isdir(item_path):
                has_mp4 = any(f.lower().endswith('.mp4') for f in os.listdir(item_path))
                if has_mp4:
                    if verbose:
                        print(f"Using lecture-downloads directory: {lecture_downloads_path}")
                    return lecture_downloads_path
    
    # Fall back to base directory
    if verbose:
        print(f"Using base directory: {base_dir}")
    return base_dir


# Public functional API
def merge_videos(
    base_dir: str = ".",
    verbose: bool = False,
    # Legacy support (auto-detected)
    input_dir: str = None,
    output_dir: str = None
) -> Dict[str, List[str]]:
    """
    Merge videos by module with chapter markers.
    
    Automatically detects user intent based on parameters:
    - If only base_dir provided: Uses new simplified interface with auto-detection
    - If input_dir and output_dir provided: Uses legacy direct paths mode
    - If only input_dir provided: Uses smart detection on input, default output location
    
    Args:
        base_dir: Base project directory (auto-detects input, outputs to base_dir/merged-lectures)
        verbose: Enable progress output
        input_dir: Legacy parameter - if provided, auto-detects direct vs smart mode
        output_dir: Legacy parameter - if provided with input_dir, uses direct paths mode
    
    Returns:
        {"successful": [...], "failed": [...]}
    """
    # Auto-detect user intent based on parameters provided
    if input_dir is not None:
        # Legacy mode detected
        if output_dir is not None:
            # Both input and output specified = check if we should use direct paths or smart detection
            # If input_dir contains module directories with MP4s, use it directly
            # Otherwise, apply smart detection
            has_modules_with_mp4s = False
            if os.path.exists(input_dir) and os.path.isdir(input_dir):
                for item in os.listdir(input_dir):
                    item_path = os.path.join(input_dir, item)
                    if os.path.isdir(item_path):
                        has_mp4 = any(f.lower().endswith('.mp4') for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f)))
                        if has_mp4:
                            has_modules_with_mp4s = True
                            break
            
            if has_modules_with_mp4s:
                # Direct paths mode - input_dir has the expected structure
                final_input_dir = input_dir
                final_output_dir = output_dir
                if verbose:
                    print(f"Using direct paths mode: {input_dir} -> {output_dir}")
            else:
                # Apply smart detection even with both parameters
                final_input_dir = _detect_input_directory(input_dir, verbose)
                final_output_dir = output_dir
                if verbose:
                    print(f"Using smart detection with explicit output: {final_input_dir} -> {output_dir}")
        else:
            # Only input specified = smart detection on input, default output
            final_input_dir = _detect_input_directory(input_dir, verbose)
            final_output_dir = os.path.join(input_dir, "merged-lectures")
            if verbose:
                print(f"Using legacy mode with smart detection: '{final_input_dir}' -> '{final_output_dir}'")
    else:
        # New simplified mode
        final_input_dir = _detect_input_directory(base_dir, verbose)
        final_output_dir = os.path.join(base_dir, "merged-lectures")
        if verbose:
            from rich.console import Console
            console = Console()
            console.print(f"[bold blue]Using simplified mode:[/bold blue] [green]'{final_input_dir}'[/green] [yellow]→[/yellow] [cyan]'{final_output_dir}'[/cyan]")
    
    if not os.path.exists(final_input_dir):
        raise FileNotFoundError(f"Input directory not found: {final_input_dir}")
    
    # Find modules to process for display
    modules_to_process = []
    for item in os.listdir(final_input_dir):
        item_path = os.path.join(final_input_dir, item)
        if os.path.isdir(item_path):
            has_mp4 = any(f.lower().endswith('.mp4') for f in os.listdir(item_path))
            if has_mp4:
                modules_to_process.append((item, item_path))
    
    # Show clean mapping
    _print_merge_mapping(modules_to_process, final_input_dir, final_output_dir, verbose)
    
    # Execute merge (handles async internally)
    return asyncio.run(_merge_all_modules_async(final_input_dir, final_output_dir, verbose))

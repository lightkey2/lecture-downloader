#!/usr/bin/env python3
"""
Download functionality for lecture_downloader package.
"""

import os
import re
import json
import asyncio
import subprocess
import nest_asyncio
from pathlib import Path
import concurrent.futures
from typing import List, Dict, Optional, Tuple, Union
nest_asyncio.apply()  # Allow nested event loops in Jupyter notebooks

from .utils import natural_sort_key, get_ffmpeg_exe


def _analyze_ffmpeg_error(stderr_output: str) -> Tuple[bool, str]:
    """
    Analyze ffmpeg stderr output to determine if it's a URL-related error.
    
    Returns:
        Tuple of (is_url_error, user_friendly_message)
    """
    stderr_lower = stderr_output.lower()
    
    # Common URL/network error patterns
    url_error_patterns = [
        # HTTP errors
        ('403', 'Access forbidden (403) - the link may have expired or requires authentication'),
        ('404', 'Resource not found (404) - the link appears to be invalid'),
        ('401', 'Unauthorized access (401) - the link may have expired or requires authentication'),
        ('500', 'Server error (500) - the video server is experiencing issues'),
        ('502', 'Bad gateway (502) - the video server is temporarily unavailable'),
        ('503', 'Service unavailable (503) - the video server is temporarily down'),
        
        # Network/connection errors
        ('connection refused', 'Connection refused - the server is not responding'),
        ('connection timed out', 'Connection timed out - the server is not responding or the link may be invalid'),
        ('network is unreachable', 'Network unreachable - check your internet connection'),
        ('no route to host', 'Cannot reach the server - the link may be invalid'),
        ('name or service not known', 'Cannot resolve server address - the link may be invalid'),
        ('temporary failure in name resolution', 'DNS resolution failed - check your internet connection'),
        
        # SSL/TLS errors
        ('ssl', 'SSL/TLS connection error - the server may have certificate issues'),
        ('certificate', 'SSL certificate error - the server may have security issues'),
        
        # Protocol errors
        ('protocol not found', 'Unsupported protocol - the link format may be invalid'),
        ('invalid url', 'Invalid URL format'),
        ('malformed', 'Malformed URL - the link appears to be corrupted'),
        
        # Stream/format errors that often indicate expired links
        ('no streams', 'No video streams found - the link may have expired'),
        ('invalid data found', 'Invalid video data - the link may have expired or be corrupted'),
        ('end of file', 'Unexpected end of file - the link may have expired'),
        ('server returned 4', 'Server returned 4xx error - the link may have expired or be invalid'),
        ('server returned 5', 'Server returned 5xx error - the video server is experiencing issues'),
    ]
    
    for pattern, message in url_error_patterns:
        if pattern in stderr_lower:
            return True, f"Cannot download video: {message}. Please check if the link is still valid."
    
    # Check for generic HTTP error codes
    import re
    http_error_match = re.search(r'http.*?(\d{3})', stderr_lower)
    if http_error_match:
        status_code = http_error_match.group(1)
        if status_code.startswith('4'):
            return True, f"Cannot download video: HTTP {status_code} error - the link may have expired or be invalid."
        elif status_code.startswith('5'):
            return True, f"Cannot download video: HTTP {status_code} error - the video server is experiencing issues."
    
    return False, ""


def _print_download_mapping(lectures: List[Dict[str, str]], output_dir: str, verbose: bool = False):
    """Print a clean tree view of the download mapping."""
    # Always show the clean tree view (removed logging dependency)
    
    # Group lectures by module
    modules = {}
    for lecture in lectures:
        if 'module' in lecture:
            module_name = lecture['module']
            if module_name not in modules:
                modules[module_name] = []
            modules[module_name].append(lecture['filename'])
        else:
            # Handle non-module lectures
            if 'Individual Lectures' not in modules:
                modules['Individual Lectures'] = []
            modules['Individual Lectures'].append(lecture['title'])
    
    print(f"Download Plan:")
    print(f"  Output: {output_dir}")
    print(f"  Modules: {len(modules)}")
    print(f"  Total Lectures: {len(lectures)}")
    print()
    
    for i, (module_name, lecture_titles) in enumerate(modules.items()):
        is_last_module = i == len(modules) - 1
        module_prefix = "└── " if is_last_module else "├── "
        print(f"{module_prefix}{module_name} ({len(lecture_titles)} lectures)")
        
        for j, title in enumerate(lecture_titles):
            is_last_lecture = j == len(lecture_titles) - 1
            if is_last_module:
                lecture_prefix = "    └── " if is_last_lecture else "    ├── "
            else:
                lecture_prefix = "│   └── " if is_last_lecture else "│   ├── "
            print(f"{lecture_prefix}{title}")
    print()


async def _download_single_lecture(lecture: Dict[str, str], base_dir: str, use_custom_titles: bool = False, verbose: bool = True) -> bool:
    """Download a single lecture using ffmpeg."""
    try:
        if use_custom_titles and 'custom_title' in lecture and 'module' in lecture:
            module_name = lecture['module']
            filename = lecture['filename']
            
            module_dir = os.path.join(base_dir, module_name)
            os.makedirs(module_dir, exist_ok=True)
            
            # Ensure filename has .mp4 extension but don't duplicate it
            if not filename.lower().endswith('.mp4'):
                filename = f"{filename}.mp4"
            save_path = os.path.join(module_dir, filename)
            display_title = f"{module_name}/{filename}"
        else:
            # Ensure title has .mp4 extension but don't duplicate it
            title = lecture['title']
            if not title.lower().endswith('.mp4'):
                title = f"{title}.mp4"
            save_path = os.path.join(base_dir, title)
            display_title = lecture['title']
            os.makedirs(base_dir, exist_ok=True)
        
        if os.path.exists(save_path):
            if verbose:
                print(f"Skipping {display_title} - file already exists")
            return True

        print(f"Downloading {display_title} to {save_path}...")
            
        process = await asyncio.create_subprocess_exec(
            get_ffmpeg_exe(),
            "-i", lecture["url"],
            "-c", "copy",
            "-bsf:a", "aac_adtstoasc",
            "-y",
            save_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            stderr_text = stderr.decode()
            
            # Analyze the error to provide better user feedback
            is_url_error, user_message = _analyze_ffmpeg_error(stderr_text)
            
            if is_url_error:
                # URL-related error - provide user-friendly message
                print(f"❌ {display_title}: {user_message}")
            else:
                # Other ffmpeg error - show technical details
                print(f"❌ {display_title}: FFmpeg error (exit code {process.returncode})")
                if verbose:
                    print(f"   Error details: {stderr_text.strip()}")
            
            return False

        print(f"Successfully downloaded: {display_title}")
        return True

    except subprocess.CalledProcessError as e:
        # Handle ffmpeg-specific errors
        stderr_text = e.stderr if hasattr(e, 'stderr') and e.stderr else str(e)
        is_url_error, user_message = _analyze_ffmpeg_error(stderr_text)
        
        if is_url_error:
            print(f"❌ {display_title}: {user_message}")
        else:
            print(f"❌ {display_title}: FFmpeg error (exit code {e.returncode})")
            if verbose:
                print(f"   Error details: {stderr_text.strip()}")
        return False
        
    except Exception as e:
        # Handle other exceptions (file system, network, etc.)
        error_str = str(e)
        is_url_error, user_message = _analyze_ffmpeg_error(error_str)
        
        if is_url_error:
            print(f"❌ {display_title}: {user_message}")
        else:
            print(f"❌ {display_title}: Unexpected error - {error_str}")
        return False


async def _download_lectures_async(
    lectures: List[Dict[str, str]], 
    base_dir: str, 
    max_workers: int = 6, 
    use_custom_titles: bool = False,
    verbose: bool = True
) -> Dict[str, List[str]]:
    """Download multiple lectures concurrently."""
    results = {"successful": [], "failed": []}
    
    if verbose:
        print(f"Starting download of {len(lectures)} lectures with {max_workers} concurrent workers")
        print(f"Output directory: {base_dir}")
        print(f"Use custom titles: {use_custom_titles}")

    # Create semaphore to limit concurrent downloads
    semaphore = asyncio.Semaphore(max_workers)
    
    async def download_with_semaphore(lecture):
        async with semaphore:
            return await _download_single_lecture(lecture, base_dir, use_custom_titles, verbose)
    
    # Create tasks for all downloads
    tasks = [download_with_semaphore(lecture) for lecture in lectures]
    
    # Execute downloads and collect results
    completed = 0
    total = len(tasks)
    
    for i, task in enumerate(asyncio.as_completed(tasks)):
        lecture = lectures[i]
        display_title = lecture.get('display_title', lecture['title'])
        
        try:
            success = await task
            if success:
                results["successful"].append(display_title)
            else:
                results["failed"].append(display_title)
        except Exception as e:
            error_msg = f"Unexpected error for {display_title}: {str(e)}"
            print(error_msg)
            results["failed"].append(display_title)
        
        completed += 1
        if verbose:
            print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")

    summary = f"Download Summary: Successful: {len(results['successful'])}, Failed: {len(results['failed'])}"
    print(summary)

    return results


def _detect_file_format(file_path: str) -> str:
    """Detect if the links file has unit delimiters or is just a list of links."""
    has_delimiters = False
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if not line.startswith('http'):
                has_delimiters = True
                break
    
    return "delimited" if has_delimiters else "sequential"


def _parse_delimited_links(file_path: str, verbose: bool = False) -> Dict[str, List[str]]:
    """Parse links file with unit delimiters (original functionality)."""
    parsed_links = {}
    current_unit = None
    
    if verbose:
        print("Detected delimited format (with unit separators)")
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
                
            if not line.startswith('http'):
                current_unit = line
                parsed_links[current_unit] = []
                if verbose:
                    print(f"Found unit: {current_unit}")
            else:
                if current_unit:
                    parsed_links[current_unit].append(line)
    
    if verbose:
        print(f"Parsed {len(parsed_links)} units with a total of {sum(len(links) for links in parsed_links.values())} links")
    
    return parsed_links


def _parse_sequential_links(file_path: str, verbose: bool = False) -> Dict[str, List[str]]:
    """Parse links file without unit delimiters (sequential format)."""
    links = []
    
    if verbose:
        print("Detected sequential format (no unit separators)")
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and line.startswith('http'):
                links.append(line)
    
    if verbose:
        print(f"Found {len(links)} links in sequential format")
    
    # Return as a single "sequential" unit for now - will be properly mapped later
    return {"sequential_links": links}


def _parse_links_file(file_path: str, verbose: bool = False) -> Dict[str, List[str]]:
    """Parse the links file to extract unit names and links, handling both delimited and sequential formats."""
    if verbose:
        print(f"Parsing links file: {file_path}")
    
    file_format = _detect_file_format(file_path)
    
    if file_format == "delimited":
        return _parse_delimited_links(file_path, verbose)
    else:
        return _parse_sequential_links(file_path, verbose)


def _parse_titles_file(file_path: str, verbose: bool = False) -> Optional[Dict[str, List[str]]]:
    """Parse a JSON file containing custom lecture titles."""
    if not file_path or not os.path.exists(file_path):
        if file_path:
            if verbose:
                print(f"Titles file not found: {file_path}")
        return None
    
    try:
        if verbose:
            print(f"Parsing titles file: {file_path}")
        
        with open(file_path, 'r') as file:
            titles_data = json.load(file)
        
        if verbose:
            print(f"Parsed {len(titles_data)} modules with custom titles")
            
        return titles_data
    except Exception as e:
        print(f"Error parsing titles file: {str(e)}")
        return None


def _parse_links_input(links: Union[str, List[str]], verbose: bool = False) -> Dict[str, List[str]]:
    """Parse links input into standardized format."""
    if isinstance(links, str):
        if links.startswith('http'):
            # Single URL
            return {"single_link": [links]}
        else:
            # File path
            return _parse_links_file(links, verbose)
    elif isinstance(links, list):
        # List of URLs
        return {"link_list": links}
    else:
        raise ValueError("Links must be a file path, single URL, or list of URLs")


def _parse_titles_input(titles, verbose: bool = False) -> Optional[Dict]:
    """Parse titles input into standardized format."""
    if titles is None:
        return None
    elif isinstance(titles, str):
        if titles.startswith('http') or not os.path.exists(titles):
            # Single title string
            return {"single_title": [titles]}
        else:
            # File path to JSON
            return _parse_titles_file(titles, verbose)
    elif isinstance(titles, list):
        # List of titles
        return {"sequential_titles": titles}
    elif isinstance(titles, dict):
        # Direct dict mapping
        return titles
    else:
        raise ValueError("Titles must be a file path, list, dict, or None")


def _map_units_to_modules(units: Dict[str, List[str]], titles: Dict[str, List[str]], verbose: bool = False) -> Dict[str, Tuple[str, List[str]]]:
    """Map unit names to module names and their corresponding titles."""
    if not titles:
        return {}
    
    # Handle single link + single title case
    if "single_link" in units and "single_title" in titles:
        if verbose:
            print("Creating direct mapping for single link + single title")
        return {"single_title": titles["single_title"]}
    
    # Check if this is sequential links (no unit delimiters)
    if "sequential_links" in units:
        return _map_sequential_links_to_modules(units["sequential_links"], titles, verbose)
    
    # Original delimited mapping logic
    def extract_unit_number(unit_name: str) -> int:
        """Extract numeric part from unit name (e.g., 'unit_10' -> 10)"""
        try:
            return int(unit_name.split('_')[1])
        except (IndexError, ValueError):
            return 0
    
    def extract_module_number(module_name: str) -> int:
        """Extract numeric part from module name (e.g., 'Module 10: Title' -> 10)"""
        try:
            # Split by ':' and take the first part, then extract the number
            module_part = module_name.split(':')[0].strip()
            return int(module_part.split()[-1])  # Get the last word which should be the number
        except (IndexError, ValueError):
            return 0
    
    # Sort units and modules by their numeric values
    sorted_units = sorted(units.keys(), key=extract_unit_number)
    sorted_modules = sorted(titles.keys(), key=extract_module_number)
    
    if len(sorted_units) > len(sorted_modules):
        print(f"Warning: More units ({len(sorted_units)}) than modules ({len(sorted_modules)})")
    
    unit_to_module = {}
    for i, unit in enumerate(sorted_units):
        if i < len(sorted_modules):
            module = sorted_modules[i]
            unit_to_module[unit] = (module, titles[module])
            if verbose:
                print(f"Mapped {unit} to {module} with {len(titles[module])} titles")
    
    return unit_to_module


def _map_sequential_links_to_modules(links: List[str], titles: Dict[str, List[str]], verbose: bool = False) -> Dict[str, Tuple[str, List[str]]]:
    """Map sequential links to modules based on lecture title counts."""
    if verbose:
        print("Mapping sequential links to modules based on lecture title counts...")
    
    def extract_module_number(module_name: str) -> int:
        """Extract numeric part from module name (e.g., 'Module 10: Title' -> 10)"""
        try:
            module_part = module_name.split(':')[0].strip()
            return int(module_part.split()[-1])
        except (IndexError, ValueError):
            return 0
    
    # Sort modules by their numeric values
    sorted_modules = sorted(titles.keys(), key=extract_module_number)
    
    # Calculate total expected lectures
    total_expected_lectures = sum(len(module_titles) for module_titles in titles.values())
    total_links = len(links)
    
    if verbose:
        print(f"Total links: {total_links}")
        print(f"Total expected lectures: {total_expected_lectures}")
    
    if total_links != total_expected_lectures:
        print(f"Warning: Link count ({total_links}) doesn't match expected lecture count ({total_expected_lectures})")
    
    # Distribute links across modules
    unit_to_module = {}
    link_index = 0
    
    for module in sorted_modules:
        module_titles = titles[module]
        module_link_count = len(module_titles)
        
        # Get the links for this module
        module_links = links[link_index:link_index + module_link_count]
        
        if len(module_links) > 0:
            # Create a synthetic unit name for this module
            synthetic_unit = f"sequential_module_{extract_module_number(module):02d}"
            unit_to_module[synthetic_unit] = (module, module_titles)
            
            if verbose:
                print(f"Mapped {len(module_links)} links to {module} ({synthetic_unit})")
            
            link_index += module_link_count
        
        # Stop if we've used all links
        if link_index >= total_links:
            break
    
    return unit_to_module


def _generate_lecture_data(parsed_links: Dict[str, List[str]], 
                          titles_mapping: Optional[Dict[str, Tuple[str, List[str]]]] = None,
                          verbose: bool = False) -> List[Dict[str, str]]:
    """Generate lecture data with titles."""
    lectures = []
    
    if verbose:
        print("Generating lecture data...")
    
    # Handle sequential links (no unit delimiters)
    if "sequential_links" in parsed_links:
        return _generate_sequential_lecture_data(parsed_links["sequential_links"], titles_mapping, verbose)
    
    # Handle single link
    if "single_link" in parsed_links:
        return _generate_single_link_data(parsed_links["single_link"], titles_mapping)
    
    # Handle link list
    if "link_list" in parsed_links:
        return _generate_link_list_data(parsed_links["link_list"], titles_mapping)
    
    # Handle delimited links (original functionality)
    for unit, links in parsed_links.items():
        module_info = titles_mapping.get(unit) if titles_mapping else None
        
        for i, link in enumerate(links, 1):
            default_title = f"{unit}_{i:02d}"
            
            lecture_data = {
                'title': default_title,
                'url': link
            }
            
            if module_info:
                module_name, module_titles = module_info
                
                if i <= len(module_titles):
                    custom_title = module_titles[i-1]
                    safe_title = custom_title.replace('/', '-').replace('\\', '-')
                    
                    lecture_data['module'] = module_name
                    lecture_data['custom_title'] = safe_title
                    lecture_data['display_title'] = safe_title
                    lecture_data['filename'] = safe_title
            
            lectures.append(lecture_data)
    
    return lectures


def _generate_single_link_data(links: List[str], titles_mapping: Optional[Dict]) -> List[Dict[str, str]]:
    """Generate lecture data for a single link."""
    lectures = []
    
    for i, link in enumerate(links, 1):
        # Determine the title to use - prioritize custom title
        print(f"titles_mapping: {titles_mapping}")
        if titles_mapping and "single_title" in titles_mapping:
            title = titles_mapping["single_title"][0] if titles_mapping["single_title"] else f"Lecture {i}"
            safe_title = title.replace('/', '-').replace('\\', '-')
        else:
            safe_title = f"single_lecture_{i:02d}"
        
        lecture_data = {
            'title': safe_title,
            'url': link,
            'display_title': safe_title,
            'filename': safe_title
        }
        
        lectures.append(lecture_data)
    
    return lectures


def _generate_link_list_data(links: List[str], titles_mapping: Optional[Dict]) -> List[Dict[str, str]]:
    """Generate lecture data for a list of links."""
    lectures = []
    
    for i, link in enumerate(links, 1):
        default_title = f"lecture_{i:02d}"
        
        lecture_data = {
            'title': default_title,
            'url': link
        }
        
        # Handle sequential titles
        if titles_mapping and "sequential_titles" in titles_mapping:
            titles_list = titles_mapping["sequential_titles"]
            if i <= len(titles_list):
                title = titles_list[i-1]
                safe_title = title.replace('/', '-').replace('\\', '-')
                # Update the main title field so it's used for filename
                lecture_data['title'] = safe_title
                lecture_data['display_title'] = safe_title
                lecture_data['filename'] = safe_title
        
        lectures.append(lecture_data)
    
    return lectures


def _generate_sequential_lecture_data(links: List[str], 
                                    titles_mapping: Optional[Dict[str, Tuple[str, List[str]]]] = None,
                                    verbose: bool = False) -> List[Dict[str, str]]:
    """Generate lecture data for sequential links."""
    lectures = []
    
    if verbose:
        print("Generating sequential lecture data...")
    
    if not titles_mapping:
        # No titles mapping, use default titles
        for i, link in enumerate(links, 1):
            lecture_data = {
                'title': f"sequential_links_{i:02d}",
                'url': link
            }
            lectures.append(lecture_data)
        return lectures
    
    # Extract module number for sorting synthetic units
    def extract_module_number(unit_name: str) -> int:
        try:
            return int(unit_name.split('_')[-1])
        except (IndexError, ValueError):
            return 0
    
    # Sort synthetic units by module number
    sorted_units = sorted(titles_mapping.keys(), key=extract_module_number)
    
    link_index = 0
    for synthetic_unit in sorted_units:
        module_name, module_titles = titles_mapping[synthetic_unit]
        module_link_count = len(module_titles)
        
        # Get links for this module
        module_links = links[link_index:link_index + module_link_count]
        
        for i, link in enumerate(module_links, 1):
            default_title = f"{synthetic_unit}_{i:02d}"
            
            lecture_data = {
                'title': default_title,
                'url': link
            }
            
            if i <= len(module_titles):
                custom_title = module_titles[i-1]
                safe_title = custom_title.replace('/', '-').replace('\\', '-')
                
                lecture_data['module'] = module_name
                lecture_data['custom_title'] = safe_title
                lecture_data['display_title'] = safe_title
                lecture_data['filename'] = safe_title
            
            lectures.append(lecture_data)
        
        link_index += module_link_count
        
        # Stop if we've used all links
        if link_index >= len(links):
            break
    
    return lectures


# Public functional API
def download_lectures(
    links: Union[str, List[str]],
    titles: Union[str, List[str], Dict[str, List[str]], None] = None,
    base_dir: str = ".",
    max_workers: int = 5,
    use_custom_titles: bool = True,
    verbose: bool = False,
    # Legacy support (auto-detected)
    output_dir: str = None
) -> Dict[str, List[str]]:
    """
    Download lectures from Canvas links.
    
    Automatically detects user intent based on parameters:
    - If only base_dir provided: Uses new simplified interface (downloads to base_dir/lecture-downloads)
    - If output_dir provided: Uses legacy direct paths mode (downloads directly to output_dir)
    
    Args:
        links: File path to links file, single URL, or list of URLs
        titles: File path to JSON titles file, list of titles, dict mapping, or None
        base_dir: Base project directory (downloads to base_dir/lecture-downloads)
        max_workers: Number of concurrent downloads
        use_custom_titles: Whether to use custom titles from titles input
        verbose: Enable progress output
        output_dir: Legacy parameter - if provided, uses direct paths mode
    
    Returns:
        {"successful": [...], "failed": [...]}
    """
    # Auto-detect user intent based on parameters provided
    if output_dir is not None:
        # Legacy mode detected - use output_dir directly (direct paths mode)
        final_output_dir = output_dir
        if verbose:
            print(f"Using legacy direct paths mode: downloads to {output_dir}")
    else:
        # New simplified mode - use base_dir/lecture-downloads
        final_output_dir = os.path.join(base_dir, "lecture-downloads")
        if verbose:
            from rich.console import Console
            console = Console()
            console.print(f"[bold blue]Using simplified mode:[/bold blue] [green]downloads to[/green] [cyan]{final_output_dir}[/cyan]")
    
    # Parse and normalize inputs
    parsed_links = _parse_links_input(links, verbose)
    parsed_titles = _parse_titles_input(titles, verbose)
    
    # Map units to modules if titles provided
    titles_mapping = None
    if parsed_titles:
        titles_mapping = _map_units_to_modules(parsed_links, parsed_titles, verbose)
    
    # Generate lecture data
    lectures = _generate_lecture_data(parsed_links, titles_mapping, verbose)
    
    # Show clean mapping
    _print_download_mapping(lectures, final_output_dir, verbose)
    
    # Execute download (handles async internally)
    return asyncio.run(_download_lectures_async(
        lectures, final_output_dir, max_workers, use_custom_titles, verbose
    ))

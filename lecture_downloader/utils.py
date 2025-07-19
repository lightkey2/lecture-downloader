#!/usr/bin/env python3
"""
Utility functions for lecture_downloader package.
"""

import os
import re
import json
import time
import shutil
import asyncio
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

try:
    import imageio_ffmpeg as ffmpeg_lib
except ImportError:
    ffmpeg_lib = None


def get_ffmpeg_exe():
    """Get FFmpeg executable path, auto-downloading if needed."""
    if ffmpeg_lib is None:
        raise RuntimeError("imageio-ffmpeg not installed. Please install with: pip install imageio-ffmpeg")
    
    try:
        return ffmpeg_lib.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError(f"Failed to get FFmpeg: {e}")


def get_ffprobe_exe():
    """Get FFprobe executable path."""
    # Try to get ffprobe from the same directory as ffmpeg
    try:
        ffmpeg_path = get_ffmpeg_exe()
        ffprobe_path = ffmpeg_path.replace('ffmpeg', 'ffprobe')
        
        # Check if ffprobe exists in the same directory
        if os.path.exists(ffprobe_path):
            return ffprobe_path
    except:
        pass
    
    # Fallback to system ffprobe if available
    system_ffprobe = shutil.which('ffprobe')
    if system_ffprobe:
        return system_ffprobe
    
    # If neither works, try to use ffmpeg for basic probing
    try:
        return get_ffmpeg_exe()  # Some operations can use ffmpeg instead of ffprobe
    except:
        raise RuntimeError("Neither FFprobe nor FFmpeg found. Please ensure imageio-ffmpeg is properly installed.")



def natural_sort_key(text: str) -> List:
    """Natural sorting key to handle mixed text and numbers properly."""
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    return [convert(c) for c in re.split(r'([0-9]+)', text)]


def format_module_name_with_padding(module_name: str) -> str:
    """Format module name with zero-padded numbers for better file ordering."""
    try:
        # Handle formats like "Module 5: Buffer Management" or "Module 10: Advanced..."
        if ':' in module_name:
            parts = module_name.split(':')
            module_part = parts[0].strip()  # "Module 5"
            title_part = parts[1].strip()   # "Buffer Management"
            
            # Extract number from module part
            words = module_part.split()
            if len(words) >= 2 and words[0].lower() == 'module':
                try:
                    module_num = int(words[1])
                    # Format with zero padding: Module 05, Module 10, etc.
                    formatted_module_part = f"Module {module_num:02d}"
                    return f"{formatted_module_part} {title_part}"
                except ValueError:
                    pass
        
        # If parsing fails, clean the name but don't add padding
        return re.sub(r'[^\w\s-]', '', module_name).strip()
        
    except Exception:
        # Fallback to original cleaning method
        return re.sub(r'[^\w\s-]', '', module_name).strip()


def get_video_duration(filepath: str) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        cmd = [
            get_ffprobe_exe(), 
            '-v', 'quiet', 
            '-print_format', 'json', 
            '-show_format', 
            filepath
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    except (subprocess.CalledProcessError, KeyError, ValueError) as e:
        print(f"Error getting duration for {filepath}: {e}")
        return 0.0


async def extract_audio_from_video(video_path: str, output_path: str, verbose: bool = False) -> bool:
    """Extract audio from video file using FFmpeg."""
    try:
        cmd = [
            get_ffmpeg_exe(), '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # 16-bit PCM
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite
            output_path
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            stderr_text = stderr.decode()
            print(f"❌ FFmpeg error extracting audio: {stderr_text.strip()}")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return False


def seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"


def words_to_srt(word_info: List[Tuple[str, float, float]], max_chars_per_line: int = 50) -> str:
    """Convert word timestamps to SRT format."""
    if not word_info:
        return ""
    
    srt_content = ""
    subtitle_index = 1
    
    current_line = ""
    line_start_time = None
    line_end_time = None
    
    for word, start_time, end_time in word_info:
        if line_start_time is None:
            line_start_time = start_time
        
        test_line = current_line + (" " if current_line else "") + word
        
        if len(test_line) > max_chars_per_line and current_line:
            srt_content += f"{subtitle_index}\n"
            srt_content += f"{seconds_to_srt_time(line_start_time)} --> {seconds_to_srt_time(line_end_time)}\n"
            srt_content += f"{current_line.strip()}\n\n"
            
            subtitle_index += 1
            current_line = word
            line_start_time = start_time
        else:
            current_line = test_line
        
        line_end_time = end_time
    
    if current_line.strip():
        srt_content += f"{subtitle_index}\n"
        srt_content += f"{seconds_to_srt_time(line_start_time)} --> {seconds_to_srt_time(line_end_time)}\n"
        srt_content += f"{current_line.strip()}\n\n"
    
    return srt_content


def words_to_transcript(word_info: List[Tuple[str, float, float]], words_per_paragraph: int = 100) -> str:
    """Convert word timestamps to full transcript in paragraph form."""
    if not word_info:
        return ""
    
    transcript = ""
    paragraph = []
    
    for i, (word, _, _) in enumerate(word_info):
        paragraph.append(word)
        
        if (len(paragraph) >= words_per_paragraph and 
            word.endswith(('.', '!', '?'))) or i == len(word_info) - 1:
            transcript += " ".join(paragraph).strip() + "\n\n"
            paragraph = []
    
    if paragraph:
        transcript += " ".join(paragraph).strip() + "\n\n"
    
    return transcript.strip()


async def inject_subtitles(video_path: str, srt_path: str, verbose: bool = False) -> bool:
    """Inject SRT subtitles into video file using FFmpeg."""
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return False
        
    if not os.path.exists(srt_path):
        print(f"SRT file not found: {srt_path}")
        return False
    
    # Create backup
    backup_path = f"{video_path}.backup"
    
    try:
        if verbose:
            print("Creating backup of original video...")
        shutil.copy2(video_path, backup_path)
        
        # Create temporary output file
        temp_output = f"{video_path}.temp.mp4"
        
        # FFmpeg command to inject subtitles
        cmd = [
            get_ffmpeg_exe(), '-i', video_path, '-i', srt_path,
            '-c', 'copy',  # Copy video and audio streams
            '-c:s', 'mov_text',  # Subtitle codec for MP4
            '-metadata:s:s:0', 'language=eng',  # Set subtitle language
            '-disposition:s:0', 'default',  # Make subtitles default
            '-y',  # Overwrite
            temp_output
        ]
        
        if verbose:
            print("Injecting subtitles into video...")
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            stderr_text = stderr.decode()
            print(f"❌ FFmpeg error injecting subtitles: {stderr_text.strip()}")
            # Restore from backup
            if os.path.exists(backup_path):
                shutil.move(backup_path, video_path)
            return False
        
        # Replace original with the new file
        os.remove(video_path)
        shutil.move(temp_output, video_path)
        
        # Remove backup since operation was successful
        os.remove(backup_path)
        
        if verbose:
            print("Subtitles successfully injected into video")
        return True
        
    except Exception as e:
        print(f"Error injecting subtitles: {e}")
        # Restore from backup if it exists
        if os.path.exists(backup_path):
            try:
                shutil.move(backup_path, video_path)
                if verbose:
                    print("Restored original video from backup")
            except:
                pass
        return False


def create_directory_structure(base_dir: str) -> Dict[str, str]:
    """Create unified directory structure for lecture processing."""
    structure = {
        'base': base_dir,
        'downloads': os.path.join(base_dir, '01_downloads'),
        'merged': os.path.join(base_dir, '02_merged'),
        'transcripts': os.path.join(base_dir, '03_transcripts')
    }
    
    # Create directories
    for dir_path in structure.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return structure


def check_dependencies():
    """Check if required tools are available (auto-download if needed)."""
    try:
        ffmpeg_path = get_ffmpeg_exe()
        ffprobe_path = get_ffprobe_exe()
        # Success - show nothing (silent operation)
    except RuntimeError as e:
        # Only show details on failure
        print(f"FFmpeg setup failed: {e}")
        try:
            # Try to show paths for debugging
            print(f"FFmpeg path attempted: {ffmpeg_path if 'ffmpeg_path' in locals() else 'Not found'}")
            print(f"FFprobe path attempted: {ffprobe_path if 'ffprobe_path' in locals() else 'Not found'}")
        except:
            pass  # Ignore errors when trying to show debug info
        raise RuntimeError(f"FFmpeg setup failed: {e}")


def detect_transcription_method() -> str:
    """Auto-detect best transcription method based on environment."""
    gcloud_vars = [
        'GOOGLE_APPLICATION_CREDENTIALS',
        'GOOGLE_CLOUD_PROJECT', 
        'GOOGLE_CLOUD_STORAGE_BUCKET'
    ]
    
    if all(os.environ.get(var) for var in gcloud_vars):
        return "gcloud"
    else:
        return "whisper"


# Check if running in Jupyter
def detect_jupyter() -> bool:
    """Detect if running in Jupyter notebook."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False

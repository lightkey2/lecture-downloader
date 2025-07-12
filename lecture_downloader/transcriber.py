#!/usr/bin/env python3
"""
Transcription functionality for lecture_downloader package.
Supports both Google Cloud Speech-to-Text and faster-whisper.
"""

import os
import re
import time
import shutil
import asyncio
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

from .utils import (
    words_to_srt, 
    inject_subtitles,
    words_to_transcript, 
    extract_audio_from_video, 
    detect_transcription_method,
)


def _print_transcribe_mapping(videos_to_transcribe: List[str], input_path: str, output_dir: str, method: str, verbose: bool = False):
    """Print a clean tree view of the transcription mapping."""
    # Always show the clean tree view (removed logging dependency)
    
    print(f"Transcription Plan:")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_dir}")
    print(f"  Method: {method}")
    print(f"  Videos: {len(videos_to_transcribe)}")
    print()
    
    # Check if this is a single file input (output_dir is same as video directory)
    is_single_file_input = False
    if len(videos_to_transcribe) == 1:
        video_dir = os.path.dirname(os.path.abspath(videos_to_transcribe[0]))
        is_single_file_input = (output_dir == video_dir)
    
    for i, video_path in enumerate(videos_to_transcribe):
        is_last_video = i == len(videos_to_transcribe) - 1
        video_prefix = "└── " if is_last_video else "├── "
        
        # Get video name without extension
        video_name = Path(video_path).stem
        print(f"{video_prefix}{video_name}")
        
        # Show output files that will be created based on input type
        if is_single_file_input:
            # Single file input - files saved directly in same directory
            if is_last_video:
                print(f"    ├── {video_name}.txt")
                print(f"    └── {video_name}.srt")
            else:
                print(f"│   ├── {video_name}.txt")
                print(f"│   └── {video_name}.srt")
        else:
            # Directory input - files saved in transcripts subdirectory structure
            if is_last_video:
                print(f"    ├── {video_name}.txt")
                print(f"    └── srt/{video_name}.srt")
            else:
                print(f"│   ├── {video_name}.txt")
                print(f"│   └── srt/{video_name}.srt")
    print()

# Google Cloud imports (optional)
try:
    from google.cloud import speech
    from google.cloud import storage
    from google.cloud.storage import transfer_manager
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False

# Faster-whisper imports (optional)
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False


class GoogleCloudTranscriber:
    """Google Cloud Speech-to-Text transcriber."""
    
    def __init__(self, bucket_name: str, project_id: str = None):
        if not GOOGLE_CLOUD_AVAILABLE:
            raise ImportError("Google Cloud libraries not available. Install with: pip install google-cloud-speech google-cloud-storage")
        
        self.bucket_name = bucket_name
        self.project_id = project_id
        
        # Initialize clients
        if project_id:
            self.speech_client = speech.SpeechClient(client_options={"quota_project_id": project_id})
            self.storage_client = storage.Client(project=project_id)
        else:
            self.speech_client = speech.SpeechClient()
            self.storage_client = storage.Client()

    async def upload_to_gcs(self, local_path: str, blob_name: str, verbose: bool = False) -> str:
        """Upload file to Google Cloud Storage."""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            file_size = os.path.getsize(local_path)
            
            if verbose:
                print(f"Uploading {file_size/1024/1024:.1f}MB file to GCS...")
            
            start_time = time.time()
            
            if file_size > 100 * 1024 * 1024:  # Use parallel upload for files > 100MB
                if verbose:
                    print("Using parallel upload for large file...")
                transfer_manager.upload_chunks_concurrently(
                    local_path, 
                    blob_name, 
                    chunk_size=25*1024*1024,  # 25MB chunks
                    max_workers=8,
                    bucket=bucket,
                    timeout=600  # 10 minute timeout
                )
            else:
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(local_path, timeout=300)
            
            upload_time = time.time() - start_time
            speed_mbps = (file_size / 1024 / 1024) / upload_time if upload_time > 0 else 0
            if verbose:
                print(f"Upload completed in {upload_time:.1f}s ({speed_mbps:.1f} MB/s)")
            
            return f"gs://{self.bucket_name}/{blob_name}"
            
        except Exception as e:
            print(f"Error uploading to GCS: {e}")
            return None

    async def transcribe_audio_gcs(self, gcs_uri: str, language_code: str = "en-US", verbose: bool = False) -> List[Tuple[str, float, float]]:
        """Transcribe audio from Google Cloud Storage using Speech-to-Text API."""
        try:
            audio = speech.RecognitionAudio(uri=gcs_uri)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code=language_code,
                enable_word_time_offsets=True,
                enable_automatic_punctuation=True,
                model="latest_long",
                use_enhanced=True,
            )
                
            if verbose:
                print("Starting transcription...")
                
            operation = self.speech_client.long_running_recognize(config=config, audio=audio)
                
            if verbose:
                print("Waiting for transcription to complete...")
            response = operation.result(timeout=7200)  # 2 hour timeout
                
            word_info = []
            for result in response.results:
                alternative = result.alternatives[0]
                    
                for word_info_item in alternative.words:
                    word = word_info_item.word
                    start_time = word_info_item.start_time.total_seconds()
                    end_time = word_info_item.end_time.total_seconds()
                    word_info.append((word, start_time, end_time))
                
            return word_info
                
        except Exception as e:
            print(f"Error during transcription: {e}")
            return []

    async def cleanup_gcs_file(self, blob_name: str, verbose: bool = False):
        """Clean up temporary GCS file."""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(blob_name)
            blob.delete()
            if verbose:
                print("Cleaned up temporary GCS audio file")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not clean up GCS file: {e}")


class WhisperTranscriber:
    """Faster-whisper local transcriber."""
    
    def __init__(self, model_size_or_path: str = "base", device: str = "auto", compute_type: str = "auto"):
        if not FASTER_WHISPER_AVAILABLE:
            raise ImportError("faster-whisper not available. Install with: pip install faster-whisper")
        
        self.model_size_or_path = model_size_or_path
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _get_model(self):
        """Get or initialize the Whisper model (cached after first use)."""
        if self._model is None:
            print(f"Loading Whisper model: {self.model_size_or_path}")
            self._model = WhisperModel(
                self.model_size_or_path, 
                device=self.device, 
                compute_type=self.compute_type
            )
        return self._model

    async def transcribe_audio_whisper(self, audio_path: str, language: str = "en", verbose: bool = False) -> List[Tuple[str, float, float]]:
        """Transcribe audio using faster-whisper."""
        try:
            if verbose:
                print("Starting Whisper transcription...")
            
            # Get model
            model = self._get_model()
            
            # Transcribe with word timestamps
            if verbose: print(f"Transcribing {audio_path} with Whisper...")
            segments, info = model.transcribe(
                audio_path, 
                language=language,
                word_timestamps=True,
                beam_size=5
            )
            
            # Extract word-level timestamps
            word_info = []
            for segment in segments:
                if hasattr(segment, 'words') and segment.words:
                    for word in segment.words:
                        word_info.append((word.word.strip(), word.start, word.end))
            
            print(f"Whisper transcription completed: {len(word_info)} words")
            return word_info
            
        except Exception as e:
            print(f"Error during Whisper transcription: {e}")
            return []


async def _transcribe_single_video(
    video_path: str, 
    output_dir: str, 
    language: str, 
    method: str, 
    inject_subtitles_flag: bool,
    verbose: bool = False,
    model_size_or_path: str = "base"
) -> bool:
    """Transcribe a single video file."""
    video_name = Path(video_path).stem
    print(f"Processing: {video_name}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract audio
        audio_path = os.path.join(temp_dir, f"{video_name}.wav")
        if verbose:
            print("Extracting audio...")
        
        if not await extract_audio_from_video(video_path, audio_path, verbose):
            return False
        
        # Transcribe based on method
        word_info = []
        
        if method == "gcloud":
            # Google Cloud transcription
            bucket_name = os.environ.get('GOOGLE_CLOUD_STORAGE_BUCKET')
            project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
            
            if not bucket_name:
                print("Error: GOOGLE_CLOUD_STORAGE_BUCKET environment variable not set")
                return False
            
            transcriber = GoogleCloudTranscriber(bucket_name, project_id)
            
            # Upload audio to GCS
            blob_name = f"audio-transcription/{video_name}_{int(time.time())}.wav"
            if verbose:
                print("Uploading audio to Google Cloud Storage...")
            
            gcs_uri = await transcriber.upload_to_gcs(audio_path, blob_name, verbose)
            if not gcs_uri:
                return False
            
            # Transcribe
            if verbose:
                print("Transcribing audio with Google Cloud...")
            word_info = await transcriber.transcribe_audio_gcs(gcs_uri, language, verbose)
            
            # Clean up GCS file
            await transcriber.cleanup_gcs_file(blob_name, verbose)
            
        elif method == "whisper":
            # Whisper transcription
            transcriber = WhisperTranscriber(model_size_or_path=model_size_or_path)
            
            # Convert language code (GCloud uses en-US, Whisper uses en)
            whisper_language = language.split('-')[0] if '-' in language else language
            
            if verbose:
                print("Transcribing audio with Whisper...")
            word_info = await transcriber.transcribe_audio_whisper(audio_path, whisper_language, verbose)
        
        else:
            print(f"Error: Unknown transcription method: {method}")
            return False
        
        if not word_info:
            print("Error: No transcription results")
            return False
        
        # Generate SRT and transcript
        if verbose:
            print("Generating SRT file and transcript...")
        srt_content = words_to_srt(word_info)
        transcript_content = words_to_transcript(word_info)
        
        # Create output directory structure
        if not output_dir:
            output_dir = os.path.dirname(os.path.abspath(video_path))
        
        # Check if we're dealing with a single file input (output_dir is same as video directory)
        video_dir = os.path.dirname(os.path.abspath(video_path))
        is_single_file_input = (output_dir == video_dir)

        
        
        if is_single_file_input:
            # Single file input - save files directly in the same directory as the video
            transcripts_dir = output_dir
            srt_path = os.path.join(transcripts_dir, f"{video_name}.srt")
            txt_path = os.path.join(transcripts_dir, f"{video_name}.txt")
        else:
            # Directory input - create transcripts subdirectory structure
            transcripts_dir = os.path.join(output_dir)
            os.makedirs(transcripts_dir, exist_ok=True)
            
            # Create SRT subdirectory within transcripts
            srt_dir = os.path.join(transcripts_dir, "srt")
            os.makedirs(srt_dir, exist_ok=True)
            
            # Save files - SRT in subdirectory, TXT in main transcripts directory
            srt_path = os.path.join(srt_dir, f"{video_name}.srt")
            txt_path = os.path.join(transcripts_dir, f"{video_name}.txt")
        
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(transcript_content)
        
        if verbose: print(f"SRT saved: {srt_path}")
        print(f"Transcript saved: {txt_path}")
        
        # Inject subtitles into video if requested
        if inject_subtitles_flag:
            if verbose:
                print("Injecting subtitles into video...")
            if await inject_subtitles(video_path, srt_path, verbose):
                print("Subtitles successfully injected into video")
            else:
                print("Warning: Failed to inject subtitles, but transcription files were saved")
        else:
            if verbose:
                print("Skipping subtitle injection")
        
        return True


async def _transcribe_videos_async(
    input_path: str,
    output_dir: str,
    language: str,
    method: str,
    max_workers: int,
    inject_subtitles_flag: bool,
    verbose: bool = False,
    model_size_or_path: str = "base"
) -> Dict[str, List[str]]:
    """Transcribe videos with concurrent processing."""
    results = {"successful": [], "failed": []}
    
    # Collect video files
    videos_to_transcribe = []
    
    if os.path.isfile(input_path) and input_path.lower().endswith('.mp4'):
        videos_to_transcribe.append(input_path)
    elif os.path.isdir(input_path):
        for file in os.listdir(input_path):
            if file.lower().endswith('.mp4'):
                videos_to_transcribe.append(os.path.join(input_path, file))
    
    if not videos_to_transcribe:
        print("Warning: No MP4 files found for transcription")
        return results
    
    # Sort videos by module number for better ordering
    def extract_module_number_from_filename(filename: str) -> int:
        """Extract module number from filename like 'Module 05 Buffer Management.mp4'"""
        try:
            match = re.search(r'Module\s+(\d+)', filename, re.IGNORECASE)
            if match:
                return int(match.group(1))
            return 999  # Put unmatched files at the end
        except (ValueError, AttributeError):
            return 999
    
    videos_to_transcribe.sort(key=lambda x: extract_module_number_from_filename(os.path.basename(x)))
    
    if verbose:
        print(f"Transcribing {len(videos_to_transcribe)} videos using {method} method")
    
    # Create semaphore to limit concurrent transcriptions
    semaphore = asyncio.Semaphore(max_workers)
    
    async def transcribe_with_semaphore(video_path):
        async with semaphore:
            return await _transcribe_single_video(
                video_path, output_dir, language, method, inject_subtitles_flag, verbose, model_size_or_path
            )
    
    # Create tasks for all transcriptions
    tasks = []
    for video_path in videos_to_transcribe:
        task = transcribe_with_semaphore(video_path)
        tasks.append((video_path, task))
    
    # Execute transcriptions and collect results
    completed = 0
    total = len(tasks)
    
    for video_path, task in tasks:
        video_name = Path(video_path).stem
        
        try:
            success = await task
            if success:
                results["successful"].append(video_name)
                print(f"Completed: {video_name}")
            else:
                results["failed"].append(video_name)
                print(f"Failed: {video_name}")
        except Exception as e:
            results["failed"].append(video_name)
            print(f"Exception processing {video_name}: {str(e)}")
        
        completed += 1
        if verbose:
            print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")
    
    # Summary
    print(f"\nTranscription Summary:")
    print(f"   Successful: {len(results['successful'])}")
    print(f"   Failed: {len(results['failed'])}")
    
    return results


def _detect_input_path(base_dir: str, verbose: bool = False) -> str:
    """
    Smart input path detection for transcription operation.
    
    Priority:
    1. base_dir/merged-lectures (if exists and has MP4 files)
    2. base_dir/lecture-downloads (if exists and has MP4 files)
    3. base_dir (fallback)
    
    Args:
        base_dir: Base directory to search in
        
    Returns:
        Path to directory/file containing videos to transcribe
    """
    # Check for merged-lectures subdirectory first
    merged_lectures_path = os.path.join(base_dir, "merged-lectures")
    if os.path.exists(merged_lectures_path):
        if os.path.isdir(merged_lectures_path):
            # Check if directory has MP4 files
            has_mp4 = any(f.lower().endswith('.mp4') for f in os.listdir(merged_lectures_path))
            if has_mp4:
                if verbose:
                    print(f"Using merged-lectures directory: {merged_lectures_path}")
                return merged_lectures_path
        elif merged_lectures_path.lower().endswith('.mp4'):
            # Single merged video file
            if verbose:
                print(f"Using merged video file: {merged_lectures_path}")
            return merged_lectures_path
    
    # Check for lecture-downloads subdirectory
    lecture_downloads_path = os.path.join(base_dir, "lecture-downloads")
    if os.path.exists(lecture_downloads_path) and os.path.isdir(lecture_downloads_path):
        # Check if it contains any directories with MP4 files or direct MP4 files
        has_mp4_files = False
        for item in os.listdir(lecture_downloads_path):
            item_path = os.path.join(lecture_downloads_path, item)
            if os.path.isdir(item_path):
                has_mp4 = any(f.lower().endswith('.mp4') for f in os.listdir(item_path))
                if has_mp4:
                    has_mp4_files = True
                    break
            elif item.lower().endswith('.mp4'):
                has_mp4_files = True
                break
        
        if has_mp4_files:
            if verbose:
                print(f"Using lecture-downloads directory: {lecture_downloads_path}")
            return lecture_downloads_path
    
    # Fall back to base directory
    if verbose:
        print(f"Using base directory: {base_dir}")
    return base_dir


# Public functional API
def transcribe_videos(
    base_dir: str = ".",
    language: str = "en-US", # Language code for transcription (en-US for GCloud, en for Whisper)
    method: str = "auto", # "auto", "google", "whisper"
    max_workers: int = 3, 
    inject_subtitles: bool = False, 
    verbose: bool = False,
    model_size_or_path: str = "base", # Whisper model size or path to custom model
    # Legacy support (auto-detected)
    input_path: str = None,
    output_dir: str = None
) -> Dict[str, List[str]]:
    """
    Transcribe videos using best available method.
    
    Automatically detects user intent based on parameters:
    - If only base_dir provided: Uses new simplified interface with auto-detection
    - If input_path and output_dir provided: Uses legacy direct paths mode
    - If only input_path provided: Uses smart detection on input, default output location
    
    Auto-detection priority:
    1. Google Cloud (if env vars set)
    2. Faster-whisper (local fallback)
    
    Args:
        base_dir: Base project directory (auto-detects input, outputs to base_dir/transcripts)
        language: Language code (en-US for GCloud, en for Whisper)
        method: "auto", "gcloud", or "whisper"
        max_workers: Concurrent transcription workers
        inject_subtitles: Inject SRT into video files
        verbose: Enable progress output
        input_path: Legacy parameter - if provided, auto-detects direct vs smart mode
        output_dir: Legacy parameter - if provided with input_path, uses direct paths mode
    
    Returns:
        {"successful": [...], "failed": [...]}
    """
    # Auto-detect user intent based on parameters provided
    if input_path is not None:
        # Legacy mode detected
        if output_dir is not None:
            # Both input and output specified = direct paths mode (no smart detection)
            final_input_path = input_path
            final_output_dir = output_dir
            if verbose:
                print(f"Using direct paths mode: {input_path} -> {output_dir}")
        else:
            # Only input specified = smart detection on input, default output
            if os.path.isdir(input_path):
                final_input_path = _detect_input_path(input_path, verbose)
            else:
                final_input_path = input_path
            final_output_dir = os.path.join(os.path.dirname(final_input_path), "transcripts")
            if verbose:
                print(f"Using legacy mode with smart detection: {final_input_path} -> {final_output_dir}")
    else:
        # New simplified mode
        final_input_path = _detect_input_path(base_dir, verbose)
        final_output_dir = os.path.join(base_dir, "transcripts")
        if verbose:
            print(f"Using simplified mode: {final_input_path} -> {final_output_dir}")
    
    if not os.path.exists(final_input_path):
        raise FileNotFoundError(f"Input path not found: {final_input_path}")
    
    # Auto-detect transcription method
    if method == "auto":
        method = detect_transcription_method()
        if verbose:
            print(f"Auto-detected transcription method: {method}")
    
    # Validate method availability
    if method == "gcloud" and not GOOGLE_CLOUD_AVAILABLE:
        print("Warning: Google Cloud libraries not available, falling back to Whisper")
        method = "whisper"
    
    if method == "whisper" and not FASTER_WHISPER_AVAILABLE:
        raise ImportError("faster-whisper not available. Install with: pip install faster-whisper")
    
    # Collect video files for display
    videos_to_transcribe = []
    if os.path.isfile(final_input_path) and final_input_path.lower().endswith('.mp4'):
        videos_to_transcribe.append(final_input_path)
    elif os.path.isdir(final_input_path):
        for file in os.listdir(final_input_path):
            if file.lower().endswith('.mp4'):
                videos_to_transcribe.append(os.path.join(final_input_path, file))
    
    # Sort videos for consistent display
    def extract_module_number_from_filename(filename: str) -> int:
        try:
            match = re.search(r'Module\s+(\d+)', filename, re.IGNORECASE)
            if match:
                return int(match.group(1))
            return 999
        except (ValueError, AttributeError):
            return 999
    
    videos_to_transcribe.sort(key=lambda x: extract_module_number_from_filename(os.path.basename(x)))
    
    # Show clean mapping
    _print_transcribe_mapping(videos_to_transcribe, final_input_path, final_output_dir, method, verbose)
    
    # Execute transcription (handles async internally)
    return asyncio.run(_transcribe_videos_async(
        final_input_path, final_output_dir, language, method, max_workers, inject_subtitles, verbose, model_size_or_path
    ))

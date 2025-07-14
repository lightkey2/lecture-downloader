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
import subprocess
import signal
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

# Watch mode imports
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

from rich.console import Console
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.layout import Layout
from rich.align import Align

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
            # Directory input - both files saved in same transcripts directory
            if is_last_video:
                print(f"    ├── {video_name}.txt")
                print(f"    └── {video_name}.srt")
            else:
                print(f"│   ├── {video_name}.txt")
                print(f"│   └── {video_name}.srt")
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
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False


def _get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', video_path
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            return float(result.stdout.strip())
        else:
            return 0.0
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return 0.0


def _collect_video_files(input_path: str, recursive: bool = False) -> List[str]:
    """Collect MP4 files from input path, optionally recursively."""
    videos = []
    
    if os.path.isfile(input_path) and input_path.lower().endswith('.mp4'):
        videos.append(input_path)
    elif os.path.isdir(input_path):
        if recursive:
            # Recursive search using os.walk
            for root, dirs, files in os.walk(input_path):
                for file in files:
                    if file.lower().endswith('.mp4'):
                        videos.append(os.path.join(root, file))
        else:
            # Top-level only
            for file in os.listdir(input_path):
                if file.lower().endswith('.mp4'):
                    videos.append(os.path.join(input_path, file))
    
    return videos


def _check_existing_transcripts(video_path: str, output_dir: str, is_single_file_input: bool) -> bool:
    """Check if transcript files already exist for a video."""
    video_name = Path(video_path).stem
    
    if is_single_file_input:
        # Single file input - files saved directly in same directory
        srt_path = os.path.join(output_dir, f"{video_name}.srt")
        txt_path = os.path.join(output_dir, f"{video_name}.txt")
    else:
        # Directory input - both files saved in same transcripts directory
        srt_path = os.path.join(output_dir, f"{video_name}.srt")
        txt_path = os.path.join(output_dir, f"{video_name}.txt")
    
    return os.path.exists(srt_path) and os.path.exists(txt_path)


def _format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def _estimate_remaining_time(processed_duration: float, total_duration: float, elapsed_time: float) -> str:
    """Estimate remaining time based on processing speed."""
    if processed_duration <= 0 or elapsed_time <= 0:
        return "calculating..."
    
    processing_speed = processed_duration / elapsed_time  # minutes of video per second of real time
    remaining_duration = total_duration - processed_duration
    
    if processing_speed > 0:
        estimated_remaining = remaining_duration / processing_speed
        return _format_time(estimated_remaining)
    else:
        return "calculating..."


class VideoWatchHandler(FileSystemEventHandler):
    """File system event handler for watching MP4 files."""
    
    def __init__(self, transcription_queue: asyncio.Queue, processed_files: set, queued_files: set, recursive: bool = False):
        super().__init__()
        self.transcription_queue = transcription_queue
        self.processed_files = processed_files
        self.queued_files = queued_files
        self.recursive = recursive
        self.pending_files = {}  # Track files being written
        
    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory and event.src_path.lower().endswith('.mp4'):
            # Only track if not already processed or queued
            abs_path = os.path.abspath(event.src_path)
            if abs_path not in self.processed_files and abs_path not in self.queued_files:
                # Start monitoring this file for completion
                self.pending_files[abs_path] = time.time()
            
    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory and event.src_path.lower().endswith('.mp4'):
            # Only track if not already processed or queued
            abs_path = os.path.abspath(event.src_path)
            if abs_path not in self.processed_files and abs_path not in self.queued_files:
                # Update last modified time
                self.pending_files[abs_path] = time.time()
    
    async def check_stable_files(self):
        """Check for files that haven't been modified recently and queue them."""
        current_time = time.time()
        stable_files = []
        
        for file_path, last_modified in list(self.pending_files.items()):
            # File is stable if not modified for 5 seconds
            if current_time - last_modified > 5:
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    # Double-check it's not already processed or queued
                    if file_path not in self.processed_files and file_path not in self.queued_files:
                        stable_files.append(file_path)
                        self.queued_files.add(file_path)
                del self.pending_files[file_path]
        
        for file_path in stable_files:
            await self.transcription_queue.put(file_path)


class WatchStatusManager:
    """Manages clean status display for watch mode."""
    
    def __init__(self, console: Console):
        self.console = console
        self.active_files = {}  # file_path -> {"status": str, "progress": int, "start_time": float}
        self.completed_files = []  # List of completed file info
        self.stats = {"successful": 0, "failed": 0, "skipped": 0}
        self.live_display = None
        
    def add_file(self, file_path: str, status: str = "Queued"):
        """Add a file to be tracked."""
        filename = os.path.basename(file_path)
        self.active_files[file_path] = {
            "filename": filename,
            "status": status,
            "progress": 0,
            "start_time": time.time(),
            "duration": 0
        }
    
    def update_file_status(self, file_path: str, status: str, progress: int = None):
        """Update file status and optionally progress."""
        if file_path in self.active_files:
            self.active_files[file_path]["status"] = status
            if progress is not None:
                self.active_files[file_path]["progress"] = progress
            self.active_files[file_path]["duration"] = time.time() - self.active_files[file_path]["start_time"]
    
    def complete_file(self, file_path: str, success: bool, final_status: str = None):
        """Mark file as completed and move to completed list."""
        if file_path in self.active_files:
            file_info = self.active_files.pop(file_path)
            file_info["success"] = success
            file_info["final_status"] = final_status or ("✅ Complete" if success else "❌ Failed")
            self.completed_files.append(file_info)
            
            if success:
                self.stats["successful"] += 1
            else:
                self.stats["failed"] += 1
    
    def skip_file(self, file_path: str):
        """Mark file as skipped."""
        filename = os.path.basename(file_path)
        self.completed_files.append({
            "filename": filename,
            "success": True,
            "final_status": "⏭️ Skipped",
            "duration": 0
        })
        self.stats["skipped"] += 1
    
    def create_status_table(self) -> Table:
        """Create the current status table."""
        table = Table(title="📹 Processing Files", show_header=True, header_style="bold blue")
        table.add_column("File", style="cyan", width=25)
        table.add_column("Status", style="yellow", width=20)
        table.add_column("Progress", width=15)
        table.add_column("Duration", justify="right", width=10)
        
        # Add active files
        for file_info in self.active_files.values():
            progress_bar = self._create_progress_bar(file_info["progress"])
            duration_str = _format_time(file_info["duration"])
            
            table.add_row(
                file_info["filename"],
                file_info["status"],
                progress_bar,
                duration_str
            )
        
        # Show recent completed files (last 3)
        recent_completed = self.completed_files[-3:] if len(self.completed_files) > 3 else self.completed_files
        for file_info in recent_completed:
            duration_str = _format_time(file_info["duration"])
            table.add_row(
                f"[dim]{file_info['filename']}[/dim]",
                f"[dim]{file_info['final_status']}[/dim]",
                "[dim]────────[/dim]",
                f"[dim]{duration_str}[/dim]"
            )
        
        return table
    
    def _create_progress_bar(self, progress: int) -> str:
        """Create a simple text progress bar."""
        if progress == 0:
            return "[dim]░░░░░░░░[/dim]  0%"
        elif progress == 100:
            return "[green]████████[/green] 100%"
        else:
            filled = int(progress / 12.5)  # 8 blocks, so 100/8 = 12.5 per block
            bar = "█" * filled + "░" * (8 - filled)
            return f"[yellow]{bar}[/yellow] {progress:3d}%"
    
    def create_summary_panel(self, watch_path: str, method: str, workers: int, recursive: bool) -> Panel:
        """Create the summary panel."""
        # Truncate long paths
        display_path = watch_path
        if len(display_path) > 60:
            display_path = "..." + display_path[-57:]
        
        summary_text = (
            f"[bold]🔍 Watch Mode Active[/bold]\n"
            f"Directory: {display_path}\n"
            f"Recursive: {'Yes' if recursive else 'No'} | Method: {method} | Workers: {workers}\n\n"
            f"📊 Session: {self.stats['successful']} completed, {self.stats['failed']} failed, {self.stats['skipped']} skipped\n\n"
            f"[dim]Press Ctrl+C to stop watching[/dim]"
        )
        
        return Panel.fit(summary_text, title="📹 Video Transcription Watcher")


class WatchModeTranscriber:
    """Watch mode transcriber that monitors directories for new MP4 files."""
    
    def __init__(self, watch_path: str, output_dir: str, language: str, method: str, 
                 max_workers: int, inject_subtitles: bool, resume: bool, recursive: bool,
                 model_size_or_path: str = "base", verbose: bool = False):
        if not WATCHDOG_AVAILABLE:
            raise ImportError("watchdog not available. Install with: pip install watchdog")
        
        self.watch_path = watch_path
        self.output_dir = output_dir
        self.language = language
        self.method = method
        self.max_workers = max_workers
        self.inject_subtitles = inject_subtitles
        self.resume = resume
        self.recursive = recursive
        self.model_size_or_path = model_size_or_path
        self.verbose = verbose
        
        self.transcription_queue = asyncio.Queue()
        self.observer = None
        self.handler = None
        self.running = False
        self.console = Console()
        
        # Track processed files to avoid reprocessing
        self.processed_files = set()
        self.queued_files = set()
        
        # Status manager for clean display
        self.status_manager = WatchStatusManager(self.console)
        
        # Shared Whisper model to avoid reloading
        self._whisper_transcriber = None
        
        # Statistics
        self.stats = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "start_time": None
        }
    
    async def start_watching(self, process_existing: bool = True):
        """Start watching for new files."""
        self.running = True
        self.stats["start_time"] = time.time()
        
        # Initialize Whisper model once if using whisper
        if self.method == "whisper":
            self.console.print("[blue]Initializing Whisper model...[/blue]")
            self._whisper_transcriber = WhisperTranscriber(model_size_or_path=self.model_size_or_path)
            # Pre-load the model
            self._whisper_transcriber._get_model()
        
        # Process existing files if requested
        if process_existing:
            await self._process_existing_files()
        
        # Set up file system watcher
        self.handler = VideoWatchHandler(self.transcription_queue, self.processed_files, self.queued_files, self.recursive)
        self.observer = Observer()
        self.observer.schedule(self.handler, self.watch_path, recursive=self.recursive)
        self.observer.start()
        
        # Start processing loop with live display
        await self._processing_loop_with_live_display()
    
    async def _processing_loop_with_live_display(self):
        """Main processing loop with clean live display."""
        semaphore = asyncio.Semaphore(self.max_workers)
        active_tasks = set()
        
        # Create layout for live display
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=8),
            Layout(name="main", ratio=1)
        )
        
        try:
            with Live(layout, console=self.console, refresh_per_second=2) as live:
                while self.running:
                    # Update display
                    layout["header"].update(self.status_manager.create_summary_panel(
                        self.watch_path, self.method, self.max_workers, self.recursive
                    ))
                    layout["main"].update(self.status_manager.create_status_table())
                    
                    # Check for stable files periodically
                    if self.handler:
                        await self.handler.check_stable_files()
                    
                    # Process queued files
                    try:
                        video_path = await asyncio.wait_for(self.transcription_queue.get(), timeout=1.0)
                        
                        # Add to status manager
                        self.status_manager.add_file(video_path, "Queued")
                        
                        # Create transcription task
                        task = asyncio.create_task(self._transcribe_video_with_status_manager(video_path, semaphore))
                        active_tasks.add(task)
                        
                        # Clean up completed tasks
                        done_tasks = {task for task in active_tasks if task.done()}
                        for task in done_tasks:
                            active_tasks.remove(task)
                            try:
                                await task  # Get result to handle any exceptions
                            except Exception as e:
                                # Error already handled in the task
                                pass
                    
                    except asyncio.TimeoutError:
                        # No new files, continue monitoring
                        continue
                    
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Stopping watch mode...[/yellow]")
        finally:
            # Wait for active tasks to complete
            if active_tasks:
                self.console.print(f"[blue]Waiting for {len(active_tasks)} active transcriptions to complete...[/blue]")
                await asyncio.gather(*active_tasks, return_exceptions=True)
            
            # Stop observer
            if self.observer:
                self.observer.stop()
                self.observer.join()
            
            self._display_final_stats()
    
    async def _transcribe_video_with_status_manager(self, video_path: str, semaphore: asyncio.Semaphore):
        """Transcribe a single video with status manager integration."""
        async with semaphore:
            abs_path = os.path.abspath(video_path)
            
            try:
                # Remove from queued files since we're processing it now
                self.queued_files.discard(abs_path)
                
                # Check if already transcribed
                is_single_file = os.path.dirname(abs_path) == self.output_dir
                if self.resume and _check_existing_transcripts(video_path, self.output_dir, is_single_file):
                    self.stats["skipped"] += 1
                    self.processed_files.add(abs_path)
                    self.status_manager.skip_file(video_path)
                    return
                
                self.stats["processed"] += 1
                self.status_manager.update_file_status(video_path, "Starting...", 0)
                
                # Create progress and status callbacks
                def update_progress(percentage):
                    self.status_manager.update_file_status(video_path, "Transcribing...", int(percentage))
                
                def update_status(message):
                    if "Extracting audio" in message:
                        self.status_manager.update_file_status(video_path, "Extracting audio", 5)
                    elif "Transcribing with Whisper" in message:
                        self.status_manager.update_file_status(video_path, "Transcribing...", 10)
                    elif "Generating transcript" in message:
                        self.status_manager.update_file_status(video_path, "Generating files", 90)
                    elif "Injecting subtitles" in message:
                        self.status_manager.update_file_status(video_path, "Injecting subtitles", 95)
                
                # Use shared Whisper transcriber if available
                if self.method == "whisper" and self._whisper_transcriber:
                    success, srt_path, txt_path = await self._transcribe_single_video_with_shared_whisper(
                        video_path, update_progress, update_status
                    )
                else:
                    success, srt_path, txt_path = await _transcribe_single_video(
                        video_path, self.output_dir, self.language, self.method, 
                        self.inject_subtitles, self.verbose, self.model_size_or_path, 
                        update_progress, update_status
                    )
                
                if success:
                    self.stats["successful"] += 1
                    self.processed_files.add(abs_path)
                    self.status_manager.complete_file(video_path, True, "✅ Complete")
                else:
                    self.stats["failed"] += 1
                    self.processed_files.add(abs_path)
                    self.status_manager.complete_file(video_path, False, "❌ Failed")
                    
            except Exception as e:
                self.stats["failed"] += 1
                self.processed_files.add(abs_path)
                self.status_manager.complete_file(video_path, False, f"❌ Error: {str(e)[:20]}...")
    
    async def _transcribe_single_video_with_shared_whisper(self, video_path: str, progress_callback, status_callback):
        """Transcribe using the shared Whisper model to avoid reloading."""
        video_name = Path(video_path).stem
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract audio
            audio_path = os.path.join(temp_dir, f"{video_name}.wav")
            status_callback("Extracting audio...")
            
            if not await extract_audio_from_video(video_path, audio_path, verbose=False):
                status_callback("❌ Failed to extract audio")
                return False, None, None
            
            # Transcribe with shared model
            whisper_language = self.language.split('-')[0] if '-' in self.language else self.language
            status_callback("Transcribing with Whisper...")
            
            word_info = await self._whisper_transcriber.transcribe_audio_whisper(
                audio_path, whisper_language, verbose=False, progress_callback=progress_callback
            )
            
            if not word_info:
                status_callback("❌ No transcription results")
                return False, None, None
            
            # Generate and save files
            status_callback("Generating transcript files...")
            srt_content = words_to_srt(word_info)
            transcript_content = words_to_transcript(word_info)
            
            # Save files
            video_dir = os.path.dirname(os.path.abspath(video_path))
            is_single_file_input = (self.output_dir == video_dir)
            
            if is_single_file_input:
                transcripts_dir = self.output_dir
            else:
                transcripts_dir = self.output_dir
                os.makedirs(transcripts_dir, exist_ok=True)
            
            srt_path = os.path.join(transcripts_dir, f"{video_name}.srt")
            txt_path = os.path.join(transcripts_dir, f"{video_name}.txt")
            
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(transcript_content)
            
            # Inject subtitles if requested
            if self.inject_subtitles:
                status_callback("Injecting subtitles...")
                if await inject_subtitles(video_path, srt_path, verbose=False):
                    status_callback("✅ Subtitles injected successfully")
                else:
                    status_callback("⚠️ Subtitle injection failed, but transcripts saved")
            
            status_callback(f"✅ Completed transcription for {video_name}")
            return True, srt_path, txt_path
    
    async def _process_existing_files(self):
        # Start processing loop with live display
        await self._processing_loop_with_live_display()
    
    async def _processing_loop_with_live_display(self):
        """Main processing loop with clean live display."""
        semaphore = asyncio.Semaphore(self.max_workers)
        active_tasks = set()
        
        # Create layout for live display
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=8),
            Layout(name="main", ratio=1)
        )
        
        try:
            with Live(layout, console=self.console, refresh_per_second=2) as live:
                while self.running:
                    # Update display
                    layout["header"].update(self.status_manager.create_summary_panel(
                        self.watch_path, self.method, self.max_workers, self.recursive
                    ))
                    layout["main"].update(self.status_manager.create_status_table())
                    
                    # Check for stable files periodically
                    if self.handler:
                        await self.handler.check_stable_files()
                    
                    # Process queued files
                    try:
                        video_path = await asyncio.wait_for(self.transcription_queue.get(), timeout=1.0)
                        
                        # Add to status manager
                        self.status_manager.add_file(video_path, "Queued")
                        
                        # Create transcription task
                        task = asyncio.create_task(self._transcribe_video_with_status_manager(video_path, semaphore))
                        active_tasks.add(task)
                        
                        # Clean up completed tasks
                        done_tasks = {task for task in active_tasks if task.done()}
                        for task in done_tasks:
                            active_tasks.remove(task)
                            try:
                                await task  # Get result to handle any exceptions
                            except Exception as e:
                                # Error already handled in the task
                                pass
                    
                    except asyncio.TimeoutError:
                        # No new files, continue monitoring
                        continue
                    
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Stopping watch mode...[/yellow]")
        finally:
            # Wait for active tasks to complete
            if active_tasks:
                self.console.print(f"[blue]Waiting for {len(active_tasks)} active transcriptions to complete...[/blue]")
                await asyncio.gather(*active_tasks, return_exceptions=True)
            
            # Stop observer
            if self.observer:
                self.observer.stop()
                self.observer.join()
            
            self._display_final_stats()
    
    async def _transcribe_video_with_status_manager(self, video_path: str, semaphore: asyncio.Semaphore):
        """Transcribe a single video with status manager integration."""
        async with semaphore:
            abs_path = os.path.abspath(video_path)
            
            try:
                # Remove from queued files since we're processing it now
                self.queued_files.discard(abs_path)
                
                # Check if already transcribed
                is_single_file = os.path.dirname(abs_path) == self.output_dir
                if self.resume and _check_existing_transcripts(video_path, self.output_dir, is_single_file):
                    self.stats["skipped"] += 1
                    self.processed_files.add(abs_path)
                    self.status_manager.skip_file(video_path)
                    return
                
                self.stats["processed"] += 1
                self.status_manager.update_file_status(video_path, "Starting...", 0)
                
                # Create progress and status callbacks
                def update_progress(percentage):
                    self.status_manager.update_file_status(video_path, "Transcribing...", int(percentage))
                
                def update_status(message):
                    if "Extracting audio" in message:
                        self.status_manager.update_file_status(video_path, "Extracting audio", 5)
                    elif "Transcribing with Whisper" in message:
                        self.status_manager.update_file_status(video_path, "Transcribing...", 10)
                    elif "Generating transcript" in message:
                        self.status_manager.update_file_status(video_path, "Generating files", 90)
                    elif "Injecting subtitles" in message:
                        self.status_manager.update_file_status(video_path, "Injecting subtitles", 95)
                
                # Use shared Whisper transcriber if available
                if self.method == "whisper" and self._whisper_transcriber:
                    success, srt_path, txt_path = await self._transcribe_single_video_with_shared_whisper(
                        video_path, update_progress, update_status
                    )
                else:
                    success, srt_path, txt_path = await _transcribe_single_video(
                        video_path, self.output_dir, self.language, self.method, 
                        self.inject_subtitles, self.verbose, self.model_size_or_path, 
                        update_progress, update_status
                    )
                
                if success:
                    self.stats["successful"] += 1
                    self.processed_files.add(abs_path)
                    self.status_manager.complete_file(video_path, True, "✅ Complete")
                else:
                    self.stats["failed"] += 1
                    self.processed_files.add(abs_path)
                    self.status_manager.complete_file(video_path, False, "❌ Failed")
                    
            except Exception as e:
                self.stats["failed"] += 1
                self.processed_files.add(abs_path)
                self.status_manager.complete_file(video_path, False, f"❌ Error: {str(e)[:20]}...")
    
    async def _transcribe_single_video_with_shared_whisper(self, video_path: str, progress_callback, status_callback):
        """Transcribe using the shared Whisper model to avoid reloading."""
        video_name = Path(video_path).stem
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract audio
            audio_path = os.path.join(temp_dir, f"{video_name}.wav")
            status_callback("Extracting audio...")
            
            if not await extract_audio_from_video(video_path, audio_path, verbose=False):
                status_callback("❌ Failed to extract audio")
                return False, None, None
            
            # Transcribe with shared model
            whisper_language = self.language.split('-')[0] if '-' in self.language else self.language
            status_callback("Transcribing with Whisper...")
            
            word_info = await self._whisper_transcriber.transcribe_audio_whisper(
                audio_path, whisper_language, verbose=False, progress_callback=progress_callback
            )
            
            if not word_info:
                status_callback("❌ No transcription results")
                return False, None, None
            
            # Generate and save files
            status_callback("Generating transcript files...")
            srt_content = words_to_srt(word_info)
            transcript_content = words_to_transcript(word_info)
            
            # Save files
            video_dir = os.path.dirname(os.path.abspath(video_path))
            is_single_file_input = (self.output_dir == video_dir)
            
            if is_single_file_input:
                transcripts_dir = self.output_dir
            else:
                transcripts_dir = self.output_dir
                os.makedirs(transcripts_dir, exist_ok=True)
            
            srt_path = os.path.join(transcripts_dir, f"{video_name}.srt")
            txt_path = os.path.join(transcripts_dir, f"{video_name}.txt")
            
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(transcript_content)
            
            # Inject subtitles if requested
            if self.inject_subtitles:
                status_callback("Injecting subtitles...")
                if await inject_subtitles(video_path, srt_path, verbose=False):
                    status_callback("✅ Subtitles injected successfully")
                else:
                    status_callback("⚠️ Subtitle injection failed, but transcripts saved")
            
            status_callback(f"✅ Completed transcription for {video_name}")
            return True, srt_path, txt_path
    
    async def _process_existing_files(self):
        """Process existing MP4 files in the watch directory."""
        existing_videos = _collect_video_files(self.watch_path, self.recursive)
        
        if existing_videos:
            self.console.print(f"[blue]Found {len(existing_videos)} existing MP4 files[/blue]")
            
            # Filter out already transcribed if resume enabled
            videos_to_process = []
            for video_path in existing_videos:
                abs_path = os.path.abspath(video_path)
                is_single_file = len(existing_videos) == 1 and os.path.dirname(abs_path) == self.output_dir
                
                if self.resume and _check_existing_transcripts(video_path, self.output_dir, is_single_file):
                    self.stats["skipped"] += 1
                    self.processed_files.add(abs_path)  # Mark as processed to avoid reprocessing
                    if self.verbose:
                        video_name = Path(video_path).stem
                        self.console.print(f"[yellow]Skipping {video_name} (already transcribed)[/yellow]")
                else:
                    videos_to_process.append(abs_path)
            
            # Queue existing files for processing and mark as queued
            for video_path in videos_to_process:
                if video_path not in self.queued_files:
                    self.queued_files.add(video_path)
                    await self.transcription_queue.put(video_path)
    
    def _display_watch_status(self):
        """Display current watch status."""
        watch_info = Panel.fit(
            f"[bold]🔍 Watch Mode Active[/bold]\n"
            f"Directory: {self.watch_path}\n"
            f"Recursive: {'Yes' if self.recursive else 'No'}\n"
            f"Method: {self.method}\n"
            f"Workers: {self.max_workers}\n"
            f"Resume: {'Yes' if self.resume else 'No'}\n\n"
            f"[dim]Press Ctrl+C to stop watching[/dim]",
            title="📹 Video Transcription Watcher"
        )
        self.console.print(watch_info)
    
    async def _processing_loop(self):
        """Main processing loop for watch mode."""
        semaphore = asyncio.Semaphore(self.max_workers)
        active_tasks = set()
        
        try:
            while self.running:
                # Check for stable files periodically
                if self.handler:
                    await self.handler.check_stable_files()
                
                # Process queued files
                try:
                    video_path = await asyncio.wait_for(self.transcription_queue.get(), timeout=1.0)
                    
                    # Create transcription task
                    task = asyncio.create_task(self._transcribe_video_with_semaphore(video_path, semaphore))
                    active_tasks.add(task)
                    
                    # Clean up completed tasks
                    done_tasks = {task for task in active_tasks if task.done()}
                    for task in done_tasks:
                        active_tasks.remove(task)
                        try:
                            await task  # Get result to handle any exceptions
                        except Exception as e:
                            self.console.print(f"[red]Task error: {e}[/red]")
                
                except asyncio.TimeoutError:
                    # No new files, continue monitoring
                    continue
                
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Stopping watch mode...[/yellow]")
        finally:
            # Wait for active tasks to complete
            if active_tasks:
                self.console.print(f"[blue]Waiting for {len(active_tasks)} active transcriptions to complete...[/blue]")
                await asyncio.gather(*active_tasks, return_exceptions=True)
            
            # Stop observer
            if self.observer:
                self.observer.stop()
                self.observer.join()
            
            self._display_final_stats()
    
    async def _transcribe_video_with_semaphore(self, video_path: str, semaphore: asyncio.Semaphore):
        """Transcribe a single video with semaphore control."""
        async with semaphore:
            video_name = Path(video_path).stem
            abs_path = os.path.abspath(video_path)
            
            try:
                # Remove from queued files since we're processing it now
                self.queued_files.discard(abs_path)
                
                # Check if already transcribed
                is_single_file = os.path.dirname(abs_path) == self.output_dir
                if self.resume and _check_existing_transcripts(video_path, self.output_dir, is_single_file):
                    self.stats["skipped"] += 1
                    self.processed_files.add(abs_path)  # Mark as processed
                    self.console.print(f"[yellow]⏭️ Skipped: {video_name} (already transcribed)[/yellow]")
                    return
                
                self.stats["processed"] += 1
                self.console.print(f"[cyan]🎬 Processing: {video_name}[/cyan]")
                
                success, srt_path, txt_path = await _transcribe_single_video(
                    video_path, self.output_dir, self.language, self.method, 
                    self.inject_subtitles, self.verbose, self.model_size_or_path
                )
                
                if success:
                    self.stats["successful"] += 1
                    self.processed_files.add(abs_path)  # Mark as successfully processed
                    self.console.print(f"[green]✅ Completed: {video_name}[/green]")
                    if srt_path and txt_path:
                        save_dir = os.path.dirname(srt_path)
                        self.console.print(f"   📄 Saved to: {save_dir}")
                else:
                    self.stats["failed"] += 1
                    self.processed_files.add(abs_path)  # Mark as processed (even if failed) to avoid retry
                    self.console.print(f"[red]❌ Failed: {video_name}[/red]")
                    
            except Exception as e:
                self.stats["failed"] += 1
                self.processed_files.add(abs_path)  # Mark as processed (even if failed) to avoid retry
                self.console.print(f"[red]❌ Error processing {video_name}: {str(e)}[/red]")
    
    def _display_final_stats(self):
        """Display final statistics."""
        elapsed = time.time() - self.stats["start_time"] if self.stats["start_time"] else 0
        
        stats_table = Table(title="📊 Watch Mode Results")
        stats_table.add_column("Metric", style="bold")
        stats_table.add_column("Count", justify="right")
        
        stats_table.add_row("Total Processed", str(self.stats["processed"]))
        stats_table.add_row("✅ Successful", str(self.stats["successful"]))
        stats_table.add_row("❌ Failed", str(self.stats["failed"]))
        stats_table.add_row("⏭️ Skipped", str(self.stats["skipped"]))
        stats_table.add_row("⏱️ Total Time", _format_time(elapsed))
        
        self.console.print(stats_table)
        self.console.print("[green]Watch mode stopped.[/green]")


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

    async def transcribe_audio_whisper(self, audio_path: str, language: str = "en", verbose: bool = False, progress_callback=None) -> List[Tuple[str, float, float]]:
        """Transcribe audio using faster-whisper with progress tracking."""
        try:
            if verbose:
                print("Starting Whisper transcription...")
            
            # Get model
            model = self._get_model()
            
            # Get audio duration for progress calculation
            audio_duration = _get_video_duration(audio_path)
            
            # Transcribe with word timestamps
            if verbose: print(f"Transcribing {audio_path} with Whisper...")
            segments, info = model.transcribe(
                audio_path, 
                language=language,
                word_timestamps=True,
                beam_size=5
            )
            
            # Extract word-level timestamps with progress updates
            word_info = []
            processed_duration = 0.0
            
            for segment in segments:
                if hasattr(segment, 'words') and segment.words:
                    for word in segment.words:
                        word_info.append((word.word.strip(), word.start, word.end))
                
                # Update progress based on segment end time
                if hasattr(segment, 'end'):
                    processed_duration = segment.end
                    if progress_callback and audio_duration > 0:
                        progress_percentage = min(100, (processed_duration / audio_duration) * 100)
                        progress_callback(progress_percentage)
            
            # Ensure we reach 100% at the end
            if progress_callback:
                progress_callback(100)
            
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
    model_size_or_path: str = "base",
    progress_callback=None,
    status_callback=None
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Transcribe a single video file with status updates."""
    video_name = Path(video_path).stem
    console = Console()
    
    # Status update helper
    def update_status(message: str):
        if status_callback:
            status_callback(message)
        elif not progress_callback:  # Only print if not using progress bars
            console.print(f"[blue]Status:[/blue] {message}")
    
    update_status(f"Starting transcription for {video_name}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract audio
        audio_path = os.path.join(temp_dir, f"{video_name}.wav")
        update_status("Extracting audio...")
        
        if not await extract_audio_from_video(video_path, audio_path, verbose=False):
            update_status("❌ Failed to extract audio")
            return False, None, None
        
        # Transcribe based on method
        word_info = []
        
        if method == "gcloud":
            # Google Cloud transcription
            bucket_name = os.environ.get('GOOGLE_CLOUD_STORAGE_BUCKET')
            project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
            
            if not bucket_name:
                update_status("❌ GOOGLE_CLOUD_STORAGE_BUCKET environment variable not set")
                return False, None, None
            
            transcriber = GoogleCloudTranscriber(bucket_name, project_id)
            
            # Upload audio to GCS
            blob_name = f"audio-transcription/{video_name}_{int(time.time())}.wav"
            update_status("Uploading to Google Cloud Storage...")
            
            gcs_uri = await transcriber.upload_to_gcs(audio_path, blob_name, verbose=False)
            if not gcs_uri:
                update_status("❌ Failed to upload to GCS")
                return False, None, None
            
            # Transcribe
            update_status("Transcribing with Google Cloud...")
            word_info = await transcriber.transcribe_audio_gcs(gcs_uri, language, verbose=False)
            
            # Clean up GCS file
            await transcriber.cleanup_gcs_file(blob_name, verbose=False)
            
        elif method == "whisper":
            # Whisper transcription
            transcriber = WhisperTranscriber(model_size_or_path=model_size_or_path)
            
            # Convert language code (GCloud uses en-US, Whisper uses en)
            whisper_language = language.split('-')[0] if '-' in language else language
            
            update_status("Transcribing with Whisper...")
            word_info = await transcriber.transcribe_audio_whisper(audio_path, whisper_language, verbose=False, progress_callback=progress_callback)
        
        else:
            update_status(f"❌ Unknown transcription method: {method}")
            return False, None, None
        
        if not word_info:
            update_status("❌ No transcription results")
            return False, None, None
        
        # Generate SRT and transcript
        update_status("Generating transcript files...")
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
            # Directory input - save both files in the same transcripts directory
            transcripts_dir = os.path.join(output_dir)
            os.makedirs(transcripts_dir, exist_ok=True)
            
            # Save both SRT and TXT files in the same directory
            srt_path = os.path.join(transcripts_dir, f"{video_name}.srt")
            txt_path = os.path.join(transcripts_dir, f"{video_name}.txt")
        
        # Save files
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(transcript_content)
        
        # Inject subtitles into video if requested
        if inject_subtitles_flag:
            update_status("Injecting subtitles...")
            if await inject_subtitles(video_path, srt_path, verbose=False):
                update_status("✅ Subtitles injected successfully")
            else:
                update_status("⚠️ Subtitle injection failed, but transcripts saved")
        
        update_status(f"✅ Completed transcription for {video_name}")
        
        return True, srt_path, txt_path


async def _transcribe_videos_async(
    input_path: str,
    output_dir: str,
    language: str,
    method: str,
    max_workers: int,
    inject_subtitles_flag: bool,
    verbose: bool = False,
    model_size_or_path: str = "base",
    resume: bool = False,
    watch: bool = False,
    recursive: bool = False
) -> Dict[str, List[str]]:
    """Transcribe videos with concurrent processing, rich progress bars, and resume functionality."""
    console = Console()
    results = {"successful": [], "failed": [], "skipped": []}
    
    # Handle watch mode
    if watch:
        if not WATCHDOG_AVAILABLE:
            console.print("[red]Error: watchdog library not available. Install with: pip install watchdog[/red]")
            return results
        
        if not os.path.isdir(input_path):
            console.print("[red]Error: Watch mode requires a directory path[/red]")
            return results
        
        # Start watch mode
        watcher = WatchModeTranscriber(
            input_path, output_dir, language, method, max_workers, 
            inject_subtitles_flag, resume, recursive, model_size_or_path, verbose
        )
        
        try:
            await watcher.start_watching(process_existing=True)
        except KeyboardInterrupt:
            pass
        
        # Return watch mode results
        return {
            "successful": [f"watch_mode_successful_{watcher.stats['successful']}"],
            "failed": [f"watch_mode_failed_{watcher.stats['failed']}"],
            "skipped": [f"watch_mode_skipped_{watcher.stats['skipped']}"]
        }
    
    # Collect video files using recursive option
    videos_to_transcribe = _collect_video_files(input_path, recursive)
    
    if not videos_to_transcribe:
        console.print("[yellow]Warning: No MP4 files found for transcription[/yellow]")
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
    
    # Check if single file input for resume logic
    is_single_file_input = len(videos_to_transcribe) == 1 and os.path.dirname(os.path.abspath(videos_to_transcribe[0])) == output_dir
    
    # Filter out already transcribed videos if resume is enabled
    videos_to_process = []
    skipped_count = 0
    
    for video_path in videos_to_transcribe:
        if resume and _check_existing_transcripts(video_path, output_dir, is_single_file_input):
            video_name = Path(video_path).stem
            results["skipped"].append(video_name)
            skipped_count += 1
            if verbose:
                console.print(f"[yellow]Skipping {video_name} (already transcribed)[/yellow]")
        else:
            videos_to_process.append(video_path)
    
    if not videos_to_process:
        console.print("[green]All videos already transcribed! Use --no-resume to re-transcribe.[/green]")
        return results
    
    # Get video durations for time estimation
    video_durations = {}
    total_duration = 0.0
    
    console.print("[blue]Analyzing video durations...[/blue]")
    for video_path in videos_to_process:
        duration = _get_video_duration(video_path)
        video_durations[video_path] = duration
        total_duration += duration
    
    # Display summary
    summary_text = (
        f"[bold]Transcription Summary[/bold]\n"
        f"Total videos: {len(videos_to_transcribe)}\n"
        f"To process: {len(videos_to_process)}\n"
        f"Skipped: {skipped_count}\n"
        f"Total duration: {_format_time(total_duration)}\n"
        f"Method: {method}\n"
    )
    
    # Add model info for whisper
    if method == "whisper":
        summary_text += f"Whisper model: {model_size_or_path}\n"
    
    summary_text += f"Workers: {max_workers}"
    
    console.print(Panel.fit(
        summary_text,
        title="📹 Video Transcription"
    ))
    
    # Create semaphore to limit concurrent transcriptions
    semaphore = asyncio.Semaphore(max_workers)
    
    # Progress tracking
    processed_duration = 0.0
    start_time = time.time()
    
    # Create progress bars
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TextColumn("ETA:"),
        TimeRemainingColumn(),
        console=console,
        expand=True
    ) as progress:
        
        # Main progress bar
        main_task = progress.add_task(
            f"[cyan]Transcribing {len(videos_to_process)} videos",
            total=len(videos_to_process)
        )
        
        # Duration-based progress bar
        duration_task = progress.add_task(
            "[green]Processing video content",
            total=total_duration
        )
        
        async def transcribe_with_progress(video_path: str) -> bool:
            nonlocal processed_duration
            
            async with semaphore:
                video_name = Path(video_path).stem
                video_duration = video_durations.get(video_path, 0.0)
                
                # Add individual video task
                video_task = progress.add_task(
                    f"[yellow]Processing {video_name}",
                    total=100
                )
                
                try:
                    # Create progress callback for Whisper transcription
                    def update_transcription_progress(percentage):
                        progress.update(video_task, completed=percentage, description=f"[yellow]Transcribing: {video_name} ({percentage:.0f}%)")
                    
                    # Create status callback for single-line updates
                    current_status = {"message": ""}
                    def update_status(message):
                        current_status["message"] = message
                        # Don't update progress bar description during transcription to avoid conflicts
                        if not message.startswith("Transcribing with Whisper"):
                            progress.update(video_task, description=f"[yellow]{message}")
                    
                    # Transcribe the video with progress and status callbacks
                    success, srt_path, txt_path = await _transcribe_single_video(
                        video_path, output_dir, language, method, inject_subtitles_flag, verbose, model_size_or_path, update_transcription_progress, update_status
                    )
                    
                    if success:
                        progress.update(video_task, completed=100, description=f"[green]✓ Completed: {video_name}")
                        processed_duration += video_duration
                        progress.update(duration_task, completed=processed_duration)
                        
                        # Show saved transcript paths
                        if srt_path and txt_path:
                            save_dir = os.path.dirname(srt_path)
                            # Use regular print to avoid Rich wrapping the path
                            print(f"📄 Saved transcripts to: {save_dir}")
                            console.print(f"   SRT: {os.path.basename(srt_path)}")
                            console.print(f"   TXT: {os.path.basename(txt_path)}")
                        
                        # Update ETA calculation
                        elapsed = time.time() - start_time
                        if elapsed > 0 and processed_duration > 0:
                            processing_speed = processed_duration / elapsed
                            remaining_duration = total_duration - processed_duration
                            if processing_speed > 0:
                                eta_seconds = remaining_duration / processing_speed
                                progress.update(duration_task, description=f"[green]Processing content (Speed: {processing_speed:.1f}x)")
                        
                        return True
                    else:
                        progress.update(video_task, description=f"[red]✗ Failed: {video_name}")
                        return False
                        
                except Exception as e:
                    progress.update(video_task, description=f"[red]✗ Error: {video_name}")
                    if verbose:
                        console.print(f"[red]Exception processing {video_name}: {str(e)}[/red]")
                    return False
                finally:
                    # Remove individual video task after completion
                    await asyncio.sleep(0.5)  # Brief pause to show completion
                    progress.remove_task(video_task)
        
        # Create and execute tasks
        tasks = []
        for video_path in videos_to_process:
            task = transcribe_with_progress(video_path)
            tasks.append((video_path, task))
        
        # Process videos and collect results
        for video_path, task in tasks:
            video_name = Path(video_path).stem
            
            try:
                success = await task
                if success:
                    results["successful"].append(video_name)
                else:
                    results["failed"].append(video_name)
            except Exception as e:
                results["failed"].append(video_name)
                if verbose:
                    console.print(f"[red]Exception processing {video_name}: {str(e)}[/red]")
            
            # Update main progress
            progress.update(main_task, advance=1)
    
    # Final summary
    total_processed = len(results["successful"]) + len(results["failed"])
    elapsed_total = time.time() - start_time
    
    summary_table = Table(title="📊 Transcription Results")
    summary_table.add_column("Status", style="bold")
    summary_table.add_column("Count", justify="right")
    summary_table.add_column("Details", style="dim")
    
    summary_table.add_row("✅ Successful", str(len(results["successful"])), "Transcribed successfully")
    summary_table.add_row("❌ Failed", str(len(results["failed"])), "Transcription failed")
    summary_table.add_row("⏭️ Skipped", str(len(results["skipped"])), "Already transcribed (resume mode)")
    summary_table.add_row("⏱️ Total Time", _format_time(elapsed_total), f"Processing speed: {processed_duration/elapsed_total:.1f}x realtime" if elapsed_total > 0 else "")
    
    console.print(summary_table)
    
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

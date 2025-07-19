#!/usr/bin/env python3
"""
Main LectureProcessor class for lecture_downloader package.
Provides the class-based API with flexible input handling.
"""

import os
import asyncio
from typing import Dict, List, Optional, Union

from rich.console import Console

from .merger import _merge_all_modules_async
from .pipeline import _process_pipeline_async
from .transcriber import _transcribe_videos_async
from .utils import check_dependencies, detect_transcription_method, detect_jupyter
from .downloader import _parse_links_input, _parse_titles_input, _map_units_to_modules, _generate_lecture_data, _download_lectures_async


class LectureProcessor:
    """
    Main class for processing lecture videos with flexible input handling.
    
    Supports multiple input formats:
    - File paths (strings)
    - Direct content (strings, lists, dicts)
    - Mixed combinations
    """
    
    def __init__(self, verbose: bool = True, interactive: bool = True):
        """
        Initialize the LectureProcessor.
        
        Args:
            verbose: Enable detailed output
            interactive: Enable user confirmations (for Jupyter/CLI)
        """
        self.verbose = verbose
        self.interactive = interactive
        self.is_jupyter = detect_jupyter()
        self.console = Console()
    
    def download_lectures(
        self,
        links: Union[str, List[str]],  # File path, single URL, or list of URLs
        titles: Union[str, List[str], Dict[str, List[str]], None] = None,  # File path, list, or dict
        base_dir: str = ".",
        max_workers: int = 5,
        use_custom_titles: bool = True,
        # Legacy support (auto-detected)
        output_dir: str = None
    ) -> Dict[str, List[str]]:
        """
        Download lectures with flexible input handling.
        
        Automatically detects user intent based on parameters:
        - If only base_dir provided: Uses new simplified interface (downloads to base_dir/lecture-downloads)
        - If output_dir provided: Uses legacy direct paths mode (downloads directly to output_dir)
        
        Args:
            links: Can be:
                - File path to links file
                - Single URL string
                - List of URL strings
            titles: Can be:
                - File path to JSON titles file
                - List of title strings
                - Dict mapping modules to title lists
                - None (use default titles)
            base_dir: Base project directory (downloads to base_dir/lecture-downloads)
            max_workers: Number of concurrent downloads
            use_custom_titles: Whether to use custom titles
            output_dir: Legacy parameter - if provided, uses direct paths mode
        
        Returns:
            {"successful": [...], "failed": [...]}
        """
        # Check dependencies
        check_dependencies()
        
        # Auto-detect user intent based on parameters provided
        if output_dir is not None:
            # Legacy mode detected - use output_dir directly (direct paths mode)
            final_output_dir = output_dir
            if self.verbose:
                print(f"Using legacy direct paths mode: downloads to '{output_dir}'")
        else:
            # New simplified mode - use base_dir/lecture-downloads
            final_output_dir = os.path.join(base_dir, "lecture-downloads")
            if self.verbose:
                self.console.print(f"[bold blue]Using simplified mode:[/bold blue] [green]downloads to[/green] [cyan]'{final_output_dir}'[/cyan]")
        
        # Parse and normalize inputs
        parsed_links = _parse_links_input(links)
        parsed_titles = _parse_titles_input(titles)
        
        # Map units to modules if titles provided
        titles_mapping = None
        if parsed_titles:
            titles_mapping = _map_units_to_modules(parsed_links, parsed_titles)
        
        # Generate lecture data
        lectures = _generate_lecture_data(parsed_links, titles_mapping)
        
        # Show preview and get confirmation if interactive
        if self.interactive:
            if not self._confirm_download(lectures):
                return {"successful": [], "failed": []}
        
        # Execute download (handles async internally)
        return asyncio.run(_download_lectures_async(
            lectures, final_output_dir, max_workers, use_custom_titles
        ))
    
    def merge_videos(
        self,
        base_dir: str = ".",
        verbose: bool = True,
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
        # Check dependencies
        check_dependencies()
        
        # Import the detection function from merger module
        from .merger import _detect_input_directory
        
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
                    if self.verbose:
                        print(f"Using direct paths mode: '{input_dir}' -> '{output_dir}'")
                else:
                    # Apply smart detection even with both parameters
                    final_input_dir = _detect_input_directory(input_dir, self.verbose)
                    final_output_dir = output_dir
                    if self.verbose:
                        print(f"Using smart detection with explicit output: '{final_input_dir}' -> '{output_dir}'")
            else:
                # Only input specified = smart detection on input, default output
                final_input_dir = _detect_input_directory(input_dir, self.verbose)
                final_output_dir = os.path.join(input_dir, "merged-lectures")
                if self.verbose:
                    print(f"Using legacy mode with smart detection: '{final_input_dir}' -> '{final_output_dir}'")
        else:
            # New simplified mode
            final_input_dir = _detect_input_directory(base_dir, self.verbose)
            final_output_dir = os.path.join(base_dir, "merged-lectures")
            if self.verbose:
                print(f"Using simplified mode: '{final_input_dir}' -> '{final_output_dir}'")
        
        if not os.path.exists(final_input_dir):
            raise FileNotFoundError(f"Input directory not found: {final_input_dir}")
        
        if self.interactive:
            if not self._confirm_merge(final_input_dir):
                return {"successful": [], "failed": []}
        
        return asyncio.run(_merge_all_modules_async(final_input_dir, final_output_dir))
    
    def transcribe_videos(
        self,
        base_dir: str = ".",
        language: str = "en-US",
        method: str = "auto",
        max_workers: int = 3,
        inject_subtitles: bool = True,
        save_txt: bool = True,
        save_srt: bool = False,
        resume: bool = False,
        watch: bool = False,
        recursive: bool = False,
        verbose: bool = True,
        model_size_or_path: str = "base",
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
        
        Args:
            base_dir: Base project directory (auto-detects input, outputs to base_dir/transcripts)
            language: Language code
            method: "auto", "gcloud", or "whisper"
            max_workers: Concurrent transcription workers
            inject_subtitles: Inject SRT into video files
            verbose: Enable progress output
            model_size_or_path: Whisper model size or path to custom model
            input_path: Legacy parameter - if provided, auto-detects direct vs smart mode
            output_dir: Legacy parameter - if provided with input_path, uses direct paths mode
        
        Returns:
            {"successful": [...], "failed": [...]}
        """
        # Check dependencies
        check_dependencies()
        
        # Import the detection function from transcriber module
        from .transcriber import _detect_input_path
        
        # Auto-detect user intent based on parameters provided
        if input_path is not None:
            # Legacy mode detected
            if output_dir is not None:
                # Both input and output specified = direct paths mode (no smart detection)
                final_input_path = input_path
                final_output_dir = output_dir
                if self.verbose:
                    print(f"Using direct paths mode: {input_path} -> {output_dir}")
            else:
                # Only input specified = smart detection on input, default output
                if os.path.isdir(input_path):
                    final_input_path = _detect_input_path(input_path)
                    final_output_dir = os.path.join(os.path.dirname(final_input_path), "transcripts")
                else:
                    # Single file input - save transcripts in same directory as the file
                    final_input_path = input_path
                    final_output_dir = os.path.dirname(final_input_path)
                if self.verbose:
                    print(f"Using legacy mode with smart detection: {final_input_path} -> {final_output_dir}")
        else:
            # New simplified mode
            final_input_path = _detect_input_path(base_dir)
            
            # Check if base_dir is actually a file path (when user passes a single file)
            if os.path.isfile(base_dir) and base_dir.lower().endswith('.mp4'):
                # Single file passed as base_dir - save transcripts in same directory as the file
                final_output_dir = os.path.dirname(base_dir)
            else:
                # Directory passed as base_dir - use transcripts subdirectory
                final_output_dir = os.path.join(base_dir, "transcripts")
            
            if self.verbose:
                self.console.print(f"[bold blue]Using simplified mode:[/bold blue] [green]{final_input_path}[/green] [yellow]→[/yellow] [cyan]{final_output_dir}[/cyan]")
        
        if not os.path.exists(final_input_path):
            raise FileNotFoundError(f"Input path not found: {final_input_path}")
        
        if method == "auto":
            method = detect_transcription_method()
        
        if self.interactive:
            if not self._confirm_transcription(final_input_path, method):
                return {"successful": [], "failed": []}
        
        return asyncio.run(_transcribe_videos_async(
            final_input_path, final_output_dir, language, method, max_workers, inject_subtitles, save_txt, save_srt, verbose, model_size_or_path, resume, watch, recursive
        ))
    
    def process_pipeline(
        self,
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
        model_size_or_path: str = "base"
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Complete pipeline with flexible input handling.
        
        Args:
            links: Links input (file path, single URL, or list)
            titles: Titles input (file path, list, dict, or None)
            output_dir: Base output directory
            max_download_workers: Concurrent downloads
            max_transcribe_workers: Concurrent transcriptions
            transcription_method: "auto", "gcloud", or "whisper"
            language: Language code
            inject_subtitles: Inject SRT into videos
            download_only: Stop after downloading
            merge_only: Only merge existing downloads
            transcribe_only: Only transcribe existing videos
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
        
        return asyncio.run(_process_pipeline_async(
            links, titles, output_dir, max_download_workers, max_transcribe_workers,
            transcription_method, language, inject_subtitles,
            download_only, merge_only, transcribe_only, self.verbose, model_size_or_path
        ))
    
    # Helper methods for interactive confirmations
    def _format_mapping_display(self, lectures: List[Dict[str, str]]) -> str:
        """Format lecture mapping for display in a tree structure."""
        output = []
        output.append("LECTURE MAPPING PREVIEW")
        output.append("=" * 50)
        
        # Group lectures by module
        modules = {}
        for lecture in lectures:
            if 'module' in lecture:
                module = lecture['module']
                if module not in modules:
                    modules[module] = []
                modules[module].append(lecture)
        
        if not modules:
            output.append("No module mappings found - using default titles")
            return "\n".join(output)
        
        # Sort modules by number
        def extract_module_number(module_name: str) -> int:
            try:
                return int(module_name.split(':')[0].strip().split()[-1])
            except (IndexError, ValueError):
                return 0
        
        sorted_modules = sorted(modules.keys(), key=extract_module_number)
        
        total_lectures = 0
        for module in sorted_modules:
            module_lectures = modules[module]
            total_lectures += len(module_lectures)
            
            output.append(f"\n{module}")
            output.append(f"   └── {len(module_lectures)} lectures")
            
            for i, lecture in enumerate(module_lectures, 1):
                filename = lecture.get('filename', lecture['title'])
                output.append(f"       {i:2d}. {filename}")
        
        output.append(f"\nSUMMARY: {len(sorted_modules)} modules, {total_lectures} total lectures")
        return "\n".join(output)
    
    def _format_merge_plan(self, input_dir: str) -> str:
        """Format merge plan for display."""
        output = []
        output.append("VIDEO MERGE PLAN")
        output.append("=" * 50)
        
        if not os.path.exists(input_dir):
            output.append(f"Input directory not found: {input_dir}")
            return "\n".join(output)
        
        modules_to_merge = []
        for item in os.listdir(input_dir):
            item_path = os.path.join(input_dir, item)
            if os.path.isdir(item_path):
                mp4_files = [f for f in os.listdir(item_path) if f.lower().endswith('.mp4')]
                if mp4_files:
                    modules_to_merge.append((item, mp4_files))
        
        if not modules_to_merge:
            output.append("No modules with MP4 files found for merging")
            return "\n".join(output)
        
        output.append(f"Input Directory: {input_dir}")
        output.append(f"Modules to merge: {len(modules_to_merge)}")
        
        total_videos = 0
        for module_name, videos in modules_to_merge:
            total_videos += len(videos)
            output.append(f"\n{module_name}")
            output.append(f"   └── {len(videos)} videos to merge")
        
        output.append(f"\nSUMMARY: {len(modules_to_merge)} modules, {total_videos} total videos")
        return "\n".join(output)
    
    def _format_transcription_plan(self, input_path: str, method: str) -> str:
        """Format transcription plan for display."""
        output = []
        output.append("TRANSCRIPTION PLAN")
        output.append("=" * 50)
        
        if not os.path.exists(input_path):
            output.append(f"Input path not found: {input_path}")
            return "\n".join(output)
        
        videos_to_transcribe = []
        
        if os.path.isfile(input_path) and input_path.lower().endswith('.mp4'):
            videos_to_transcribe.append(input_path)
        elif os.path.isdir(input_path):
            for file in os.listdir(input_path):
                if file.lower().endswith('.mp4'):
                    videos_to_transcribe.append(os.path.join(input_path, file))
        
        if not videos_to_transcribe:
            output.append("No MP4 files found for transcription")
            return "\n".join(output)
        
        output.append(f"Input Path: {input_path}")
        output.append(f"Method: {method}")
        
        if method == "gcloud":
            bucket_name = os.environ.get('GOOGLE_CLOUD_STORAGE_BUCKET')
            project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
            if bucket_name:
                output.append(f"GCS Bucket: {bucket_name}")
            if project_id:
                output.append(f"Project ID: {project_id}")
        
        output.append(f"Videos to transcribe: {len(videos_to_transcribe)}")
        
        for i, video_path in enumerate(videos_to_transcribe, 1):
            video_name = os.path.basename(video_path)
            output.append(f"   {i:2d}. {video_name}")
        
        output.append(f"\nSUMMARY: {len(videos_to_transcribe)} videos will be transcribed")
        return "\n".join(output)
    
    def _get_user_confirmation(self, message: str, plan_text: str) -> bool:
        """Get user confirmation with different methods for CLI vs Jupyter."""
        if not self.interactive:
            return True
        
        if self.is_jupyter:
            return self._get_jupyter_confirmation(message, plan_text)
        else:
            return self._get_cli_confirmation(message, plan_text)
    
    def _get_cli_confirmation(self, message: str, plan_text: str) -> bool:
        """Get confirmation via CLI input."""
        print("\n" + plan_text)
        print("\n" + "=" * 50)
        print(f"{message}")
        
        while True:
            response = input("Continue? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no.")
    
    def _get_jupyter_confirmation(self, message: str, plan_text: str) -> bool:
        """Get confirmation via Jupyter widgets."""
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
            
            # Display the plan
            print(plan_text)
            print("\n" + "=" * 50)
            print(f"{message}")
            
            # Create confirmation widget
            confirm_button = widgets.Button(
                description="Continue",
                button_style='success',
                layout=widgets.Layout(width='120px')
            )
            
            cancel_button = widgets.Button(
                description="Cancel",
                button_style='danger',
                layout=widgets.Layout(width='120px')
            )
            
            button_box = widgets.HBox([confirm_button, cancel_button])
            
            result = {'confirmed': None}
            
            def on_confirm_clicked(b):
                result['confirmed'] = True
                clear_output(wait=True)
                print("Confirmed! Proceeding...")
            
            def on_cancel_clicked(b):
                result['confirmed'] = False
                clear_output(wait=True)
                print("Cancelled by user.")
            
            confirm_button.on_click(on_confirm_clicked)
            cancel_button.on_click(on_cancel_clicked)
            
            display(button_box)
            
            # Wait for user input
            import time
            while result['confirmed'] is None:
                time.sleep(0.1)
            
            return result['confirmed']
        except Exception as e:
            # Fallback to CLI confirmation if widgets don't work
            if self.verbose:
                print(f"Warning: Jupyter widgets not working, falling back to CLI confirmation: {e}")
            return self._get_cli_confirmation(message, plan_text)
    
    def _confirm_download(self, lectures: List[Dict[str, str]]) -> bool:
        """Get confirmation for download operation."""
        mapping_display = self._format_mapping_display(lectures)
        return self._get_user_confirmation(
            "Do you want to proceed with downloading these lectures?",
            mapping_display
        )
    
    def _confirm_merge(self, input_dir: str) -> bool:
        """Get confirmation for merge operation."""
        merge_plan = self._format_merge_plan(input_dir)
        return self._get_user_confirmation(
            "Do you want to proceed with merging these videos?",
            merge_plan
        )
    
    def _confirm_transcription(self, input_path: str, method: str) -> bool:
        """Get confirmation for transcription operation."""
        transcription_plan = self._format_transcription_plan(input_path, method)
        return self._get_user_confirmation(
            "Do you want to proceed with transcribing these videos?",
            transcription_plan
        )

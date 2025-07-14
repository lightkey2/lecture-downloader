#!/usr/bin/env python3
"""
CLI interface for lecture_downloader package.
"""

import os
import click
from .processor import LectureProcessor


@click.group()
@click.option('--verbose/--quiet', default=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose):
    """Lecture Downloader - Download, merge, and transcribe lecture videos."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose


@cli.command()
@click.option('--links', '-l', required=True, help='Path to links file or single URL')
@click.option('--titles', '-t', default=None, help='Path to titles JSON file')
@click.option('--base-dir', '-b', default='.', help='Base project directory (downloads to base-dir/lecture-downloads)')
@click.option('--max-workers', '-w', default=5, type=int, help='Concurrent downloads')
@click.option('--no-custom-titles', is_flag=True, help='Do not use custom titles')
# Legacy options (auto-detected)
@click.option('--output-dir', '-o', default=None, help='Legacy: Output directory (if provided, uses direct paths mode)')
@click.pass_context
def download(ctx, links, titles, base_dir, max_workers, no_custom_titles, output_dir):
    """Download lectures from URLs."""
    verbose = ctx.obj['verbose']
    
    processor = LectureProcessor(verbose=verbose, interactive=False)
    
    try:
        results = processor.download_lectures(
            links=links,
            titles=titles,
            base_dir=base_dir,
            max_workers=max_workers,
            use_custom_titles=not no_custom_titles,
            output_dir=output_dir
        )
        
        click.echo(f"✅ Download completed!")
        click.echo(f"   Successful: {len(results['successful'])}")
        click.echo(f"   Failed: {len(results['failed'])}")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--base-dir', '-b', default='.', help='Base project directory (auto-detects input, outputs to base-dir/merged-lectures)')
# Legacy options (auto-detected)
@click.option('--input-dir', '-i', default=None, help='Legacy: Input directory (auto-detects direct vs smart mode)')
@click.option('--output-dir', '-o', default=None, help='Legacy: Output directory (if provided with input-dir, uses direct paths mode)')
@click.pass_context
def merge(ctx, base_dir, input_dir, output_dir):
    """Merge videos by module with chapter markers."""
    verbose = ctx.obj['verbose']
    
    processor = LectureProcessor(verbose=verbose, interactive=False)
    
    try:
        results = processor.merge_videos(
            base_dir=base_dir,
            input_dir=input_dir,
            output_dir=output_dir
        )
        
        click.echo(f"✅ Merge completed!")
        click.echo(f"   Successful: {len(results['successful'])}")
        click.echo(f"   Failed: {len(results['failed'])}")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--base-dir', '-b', default='.', help='Base project directory (auto-detects input, outputs to base-dir/transcripts)')
@click.option('--method', '-m', default='auto', type=click.Choice(['auto', 'gcloud', 'whisper']), help='Transcription method')
@click.option('--language', '-lang', default='en-US', help='Language code')
@click.option('--max-workers', '-w', default=3, type=int, help='Concurrent workers')
@click.option('--no-inject', is_flag=True, help='Skip subtitle injection')
@click.option('--resume', is_flag=True, help='Skip already-transcribed files')
@click.option('--watch', is_flag=True, help='Watch directory for new MP4 files and auto-transcribe')
@click.option('--recursive', '-r', is_flag=True, help='Find MP4 files in subdirectories recursively')
# Legacy options (auto-detected)
@click.option('--input-path', '-i', default=None, help='Legacy: Video file or directory (auto-detects direct vs smart mode)')
@click.option('--output-dir', '-o', default=None, help='Legacy: Output directory (if provided with input-path, uses direct paths mode)')
@click.pass_context
def transcribe(ctx, base_dir, method, language, max_workers, no_inject, resume, watch, recursive, input_path, output_dir):
    """Transcribe videos using Google Cloud or Whisper."""
    verbose = ctx.obj['verbose']
    
    processor = LectureProcessor(verbose=verbose, interactive=False)
    
    try:
        results = processor.transcribe_videos(
            base_dir=base_dir,
            language=language,
            method=method,
            max_workers=max_workers,
            inject_subtitles=not no_inject,
            resume=resume,
            watch=watch,
            recursive=recursive,
            input_path=input_path,
            output_dir=output_dir
        )
        
        click.echo(f"✅ Transcription completed!")
        click.echo(f"   Successful: {len(results['successful'])}")
        click.echo(f"   Failed: {len(results['failed'])}")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--links', '-l', required=True, help='Path to links file')
@click.option('--titles', '-t', default=None, help='Path to titles JSON file')
@click.option('--output-dir', '-o', default='lecture_processing', help='Base output directory')
@click.option('--max-download-workers', default=5, type=int, help='Concurrent downloads')
@click.option('--max-transcribe-workers', default=3, type=int, help='Concurrent transcriptions')
@click.option('--method', '-m', default='auto', type=click.Choice(['auto', 'gcloud', 'whisper']), help='Transcription method')
@click.option('--language', '-lang', default='en-US', help='Language code')
@click.option('--download-only', is_flag=True, help='Only download')
@click.option('--merge-only', is_flag=True, help='Only merge')
@click.option('--transcribe-only', is_flag=True, help='Only transcribe')
@click.option('--no-inject', is_flag=True, help='Skip subtitle injection')
@click.pass_context
def pipeline(ctx, links, titles, output_dir, max_download_workers, max_transcribe_workers, 
             method, language, download_only, merge_only, transcribe_only, no_inject):
    """Run the complete pipeline: download -> merge -> transcribe."""
    verbose = ctx.obj['verbose']
    
    # Validate mutually exclusive options
    exclusive_flags = [download_only, merge_only, transcribe_only]
    if sum(exclusive_flags) > 1:
        click.echo("❌ Error: Only one of --download-only, --merge-only, --transcribe-only can be used", err=True)
        raise click.Abort()
    
    processor = LectureProcessor(verbose=verbose, interactive=False)
    
    try:
        results = processor.process_pipeline(
            links=links,
            titles=titles,
            output_dir=output_dir,
            max_download_workers=max_download_workers,
            max_transcribe_workers=max_transcribe_workers,
            transcription_method=method,
            language=language,
            inject_subtitles=not no_inject,
            download_only=download_only,
            merge_only=merge_only,
            transcribe_only=transcribe_only
        )
        
        click.echo(f"🎉 Pipeline completed!")
        for step, result in results.items():
            if result['successful'] or result['failed']:
                click.echo(f"   {step.title()}: {len(result['successful'])} successful, {len(result['failed'])} failed")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    cli()

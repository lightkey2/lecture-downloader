# %% [markdown]
#
#  # Lecture Downloader - Usage Guide
#
#  This notebook demonstrates the lecture_downloader package with:
#  - Complete pipeline workflow (recommended)
#  - Individual functional API operations
#  - Environment setup and configuration

# %%
import os
import lecture_downloader as ld
from lecture_downloader import LectureProcessor

#%%
# Paste links from video downloadhelper extension or video download urls here
links_content = """
https://cfvod.kaltura.com/scf/...
https://cfvod.kaltura.com/scf/...
"""
with open("example_links.txt", "w") as f:
    f.write(links_content)

# Create example titles JSON (optional if you want to use custom titles)
titles_content = """
{
  "Module 1: Introduction to ML": [
    "Database Overview_Introduction",
    "Database Overview_Fundamentals"
  ],
  "Module 2: SQL": [
    "Database Overview_Advanced Queries"
  ]
}
"""
with open("example_titles.json", "w") as f:
    f.write(titles_content)

# %% [markdown]
# ## Complete Pipeline
#
#  Run the entire workflow in a single command - download, merge, and transcribe:

# %%
processor = LectureProcessor(
    verbose=True,  # False for quiet operation
    interactive=False  # True for interactive confirmations
)

# Complete pipeline - handles everything automatically
pipeline_results = ld.process_pipeline(
    links="example_links.txt",  # or ["url1", "url2"], Can also be: single URL string, list of URLs
    titles="example_titles.json",  # or ["Title 1", "Title 2"], Can also be: single title string, list of titles, dict
    output_dir="downloaded_lectures",  # Creates organized subdirectories
    max_download_workers=8,  # 1-10, adjust based on system
    max_transcribe_workers=4,  # 1-5, adjust based on system
    transcription_method="whisper",  # "auto", "gcloud", "whisper"
    language="en",  # "en" for Whisper, "en-US" for Google Cloud
    inject_subtitles=True,  # False to skip subtitle injection
)

print("Pipeline Results:")
for step, results in pipeline_results.items():
    print(f"  {step}: {len(results['successful'])} successful, {len(results['failed'])} failed")

# %% [markdown]
#
#
#  ## Individual Operations - Functional API
#
#  Use individual functions for more control over each step:

# %%
# Single base directory where all downloads will be stored
base_dir = "Lecture-Downloads-Output"

# Step 1: Download lectures
results = ld.download_lectures(
    links="example_links.txt",  # Can also be: single URL string, list of URLs
    titles="example_titles.json",  # Can also be: single title string, list of titles, dict
    base_dir=base_dir,
    max_workers=6,  # 1-10, concurrent downloads
    verbose=False  # True for detailed logging output
)
print(f"Download: {len(results['successful'])} successful, {len(results['failed'])} failed")

# %% [markdown]
#
#
#  ### Step 2: Merge Videos

# %%
# Merge videos by module with chapter markers
merged = ld.merge_videos(base_dir=base_dir  # Auto-detects input/output directories
                         )
print(f"Merge: {len(merged['successful'])} successful, {len(merged['failed'])} failed")

# %% [markdown]
#
#
#  ### Step 3: Transcribe Videos

# %%
# Transcribe videos with automatic method detection
transcripts = ld.transcribe_videos(
    base_dir=base_dir,
    method="whisper",  # "auto", "gcloud", "whisper" - "auto" by default
    max_workers=4,  # 1-5, concurrent transcriptions
    language="en",  # "en" for Whisper, "en-US" for Google Cloud
    inject_subtitles=True  # False to skip subtitle injection into videos
)
print(f"Transcribe: {len(transcripts['successful'])} successful, {len(transcripts['failed'])} failed")

# %% [markdown]
#
#
#  ## CLI Usage Examples
#
#  ### Complete Pipeline
#  ```bash
#  # Single command for everything
#  lecture-downloader pipeline -l links.txt -t titles.json -o output
#  ```
#
#  ### Individual Operations
#  ```bash
#  # Step by step
#  BASE_DIR="AI-Course"
#  lecture-downloader download -l links.txt -t titles.json -b $BASE_DIR
#  lecture-downloader merge -b $BASE_DIR
#  lecture-downloader transcribe -b $BASE_DIR -m whisper
#  ```
#
#  ### Pipeline Options
#  ```bash
#  # Download only
#  lecture-downloader pipeline -l links.txt --download-only
#
#  # Custom settings
#  lecture-downloader pipeline \
#    -l links.txt -t titles.json -o output \
#    --max-download-workers 8 \
#    --max-transcribe-workers 4 \
#    --method whisper --language en
#  ```

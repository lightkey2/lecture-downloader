# Lecture Downloader

A Python toolkit for downloading, merging, transcribing, and embedding subtitles from lecture videos hosted on platforms like Canvas and Brightspace. 

Use at your own risk. This tool is designed for educational purposes and should not be used to violate any terms of service or copyright laws.

## Quick Start

### 30-Second Setup
```bash
# Install FFmpeg (required for video processing)
brew install ffmpeg  # macOS
# sudo apt install ffmpeg  # Ubuntu/Debian
# Windows: https://www.wikihow.com/Install-FFmpeg-on-Windows

# Install lecture downloader
pip install lecture-downloader
```
## Obtaining Video URLs from Canvas/Brightspace

Implementation based off [this reddit post](https://www.reddit.com/r/VirginiaTech/comments/13l6983/how_to_download_videos_from_canvas/)

### Using Video DownloadHelper Extension

1. **Install Extension**: Download [VideoDownloadHelper](https://www.downloadhelper.net/)
2. **Navigate to Video**: Go to your lecture video in Canvas/Brightspace
3. **Start Playback**: Click play to begin streaming
4. **Extract URL**: Click the extension icon (should be colored, not grey)
6. **Copy URL**: Click the three dots → "Copy URL"

For example, visiit [Public Lecture sample](https://cdnapisec.kaltura.com/html5/html5lib/v2.82.1/mwEmbedFrame.php/p/2019031/uiconf_id/40436601?wid=1_la8dzpbb&iframeembed=true&playerId=kaltura_player_5b490d253ef1c&flashvars%5BplaylistAPI.kpl0Id%5D=1_9hckzp35&flashvars%5BplaylistAPI.autoContinue%5D=true&flashvars%5BplaylistAPI.autoInsert%5D=true&flashvars%5Bks%5D=&flashvars%5BlocalizationCode%5D=en&flashvars%5BimageDefaultDuration%5D=30&flashvars%5BleadWithHTML5%5D=true&flashvars%5BforceMobileHTML5%5D=true&flashvars%5BnextPrevBtn.plugin%5D=true&flashvars%5BsideBarContainer.plugin%5D=true&flashvars%5BsideBarContainer.position%5D=left&flashvars%5BsideBarContainer.clickToClose%5D=true&flashvars%5Bchapters.plugin%5D=true&flashvars%5Bchapters.layout%5D=vertical&flashvars%5Bchapters.thumbnailRotator%5D=false&flashvars%5BstreamSelector.plugin%5D=true&flashvars%5BEmbedPlayer.SpinnerTarget%5D=videoHolder&flashvars%5BdualScreen.plugin%5D=true), click play on a video, and copy the URL from the extension. To bulk download, paste it into a text file named `lecture_links.txt`, one URL per line.

<img src="images/65843ae08547dc26daa123c8dd3096dace4a87ccd7643a805a57550fda5e5a14.png" width="500" alt="Lecture Downloader screenshot">
  
<img src="images/9596a4b8afd26ddc068e3160cdce0ec02a1dc22b5b512e5e1ffabf5f06d48749.png" width="300" alt="Lecture Downloader screenshot">

### Basic Usage

### One-Command Pipeline
```python
# Complete pipeline: download → merge → transcribe

pipeline_results = ld.process_pipeline(
    links="lecture_links.txt",  # Can also use: single URL string, ["url1", "url2"]
    titles="lecture_titles.json",  # Can also use: ["Title 1", "Title 2"], {"Module 1": ["Lecture 1"]}
    output_dir="lecture_processing",
    inject_subtitles=True,          # False to skip subtitle injection
    transcription_method="whisper", # "auto", "gcloud", "whisper"
    language="en",                  
)
```

### Step-by-Step Commands
```python
import lecture_downloader as ld

# Complete workflow in 3 commands
base_dir = "Lecture-Downloads"

# 1. Download lectures
results = ld.download_lectures(
    links="lecture_links.txt",  # Can also use: single URL string, ["url1", "url2"]
    titles="lecture_titles.json",  # Can also use: ["Title 1", "Title 2"], {"Module 1": ["Lecture 1"]}
    base_dir=base_dir,  # Creates Lecture-Downloads/lecture-downloads/
)

# 2. Merge videos by module with chapters
merged = ld.merge_videos(
    base_dir=base_dir,  # Auto-detects input from lecture-downloads/
)

# 3. Transcribe with Whisper (local, no setup required)
transcripts = ld.transcribe_videos(
    base_dir=base_dir,  # Auto-detects input from merged-lectures/
    method="whisper",  # "auto" detects best available method
    language="en",  # Language code for Whisper
    inject_subtitles=True,  # False to skip subtitle injection
)
```

## Installation

```bash
# Basic installation
pip install lecture-downloader
```

**Required Dependencies:**
- `ffmpeg` - Install via package manager (brew, apt, etc.)
- Python 3.8+

## Configuration Options

### Download Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `links` | str/list | Required | File path, single URL, or list of URLs |
| `titles` | str/list/dict | None | File path, list, or dict mapping |
| `base_dir` | str | "." | Base directory (creates subdirectories) |
| `max_workers` | int | 5 | Concurrent downloads (1-10) |
| `verbose` | bool | False | Detailed progress output |

### Merge Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_dir` | str | "." | Base directory (auto-detects input) |
| `verbose` | bool | False | Detailed progress output |

### Transcribe Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_dir` | str | "." | Base directory (auto-detects input) |
| `method` | str | "auto" | "auto", "whisper" |
| `language` | str | "en" | Language code for Whisper |
| `max_workers` | int | 3 | Concurrent transcriptions (1-5) |
| `inject_subtitles` | bool | True | Inject SRT into video files |
| `verbose` | bool | False | Detailed progress output |

## Input Formats

### Links Input
```python
# File with URLs (one per line)
links = "lecture_links.txt"

# Single URL
links = "https://example.com/lecture.mp4"

# List of URLs
links = ["https://url1.mp4", "https://url2.mp4"]
```

### Titles Input
```python
# JSON file with module structure
titles = "lecture_titles.json"
**lecture_links.txt:**
```
https://example.com/lecture1.mp4
https://example.com/lecture2.mp4
```
# List of titles (matches link order)
titles = ["Lecture 1", "Lecture 2", "Lecture 3"]

# Dictionary mapping modules to lectures
titles = {
    "Module 1: Introduction": ["Lecture 1", "Lecture 2"],
    "Module 2: Advanced": ["Lecture 3", "Lecture 4"]}
```
**lecture_titles.json:**
```json
{ "Module 1: Introduction": [ "Lecture 1: Overview", "Lecture 2: Fundamentals"], 
  "Module 2: Advanced Topics": [  "Lecture 3: Advanced Concepts"]}
```


## CLI Usage

### Quick Commands
```bash
# Complete workflow
BASE_DIR="Lecture-Downloads"
lecture-downloader download -l links.txt -t titles.json -b $BASE_DIR
lecture-downloader merge -b $BASE_DIR
lecture-downloader transcribe -b $BASE_DIR

# One-command pipeline
lecture-downloader pipeline -l links.txt -t titles.json -o output
```

### CLI Options
```bash
# Download with options
lecture-downloader download \
  -l links.txt \
  -t titles.json \
  -b Lecture-Downloads \
  --max-workers 8 \
  --verbose

# Transcribe with options
lecture-downloader transcribe \
  -b Lecture-Downloads \
  --method whisper \
  --language en \
  --max-workers 4 \
  --no-inject
```




**FFmpeg not found:**
```bash
# Install FFmpeg
brew install ffmpeg  # macOS
sudo apt install ffmpeg  # Ubuntu/Debian
```

### Debug Mode
```python
# Enable verbose output for troubleshooting
ld.download_lectures(links, titles, verbose=True)
ld.merge_videos(base_dir="course", verbose=True)
ld.transcribe_videos(base_dir="course", verbose=True)
```

## License

MIT License - see LICENSE file for details.
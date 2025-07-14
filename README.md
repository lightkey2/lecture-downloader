# Lecture Downloader

A Python toolkit for transcribing video and audio files using Whisper AI.

## Installation

```bash
pip install lecture-downloader
```

**Requirements:**
- Python 3.8+
- FFmpeg (automatically downloaded on first use)

## Usage

### Transcribe a Single File
```bash
# Basic transcription
lecture-downloader transcribe video.mp4

# Specify output directory
lecture-downloader transcribe video.mp4 -o transcripts/

# Choose language
lecture-downloader transcribe video.mp4 --language es
```

### Transcribe a Directory
```bash
# Transcribe all videos in a directory
lecture-downloader transcribe /path/to/videos/

# Transcribe with custom output location
lecture-downloader transcribe /path/to/videos/ -o /path/to/transcripts/
```

### Options
| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output` | Output directory for transcripts | Same as input |
| `--language` | Language code (en, es, fr, etc.) | `en` |
| `--method` | Transcription method | `whisper` |
| `--inject` | Inject subtitles into video files | `True` |
| `--verbose` | Show detailed progress | `False` |

### Examples
```bash
# Transcribe with all options
lecture-downloader transcribe lectures/ \
  --output transcripts/ \
  --language en \
  --inject \
  --verbose

# Transcribe without injecting subtitles
lecture-downloader transcribe video.mp4 --no-inject
```

## Requirements

- Python 3.8+
- FFmpeg (automatically downloaded on first use)

## License

MIT License - see LICENSE file for details.

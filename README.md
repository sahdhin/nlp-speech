# NLP Speech-to-Text with Speaker Diarization

A real-time speech transcription pipeline that captures live microphone audio, detects individual speakers, and transcribes speech to text.

## Overview

This module is the first stage of a larger NLP pipeline that:
1. Records live audio in 5-second chunks
2. Detects and labels individual speakers using pyannote.audio
3. Transcribes speech to text using OpenAI Whisper
4. Outputs labelled transcript lines to a downstream text sanitization module

## Models Used

- **OpenAI Whisper (tiny)** — Automatic Speech Recognition
- **pyannote/speaker-diarization-3.1** — Speaker Diarization

## Output Format
[SPEAKER_00]: raw transcribed text
[SPEAKER_01]: raw transcribed text

## Setup
1. Clone this repo
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your HuggingFace token: HF_TOKEN=your_token_here
4. Accept model terms at:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

## Run
python speech_to_text.py

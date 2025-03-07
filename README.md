# Image Morphing Video Generator

This program creates morphing videos between images with synchronized audio. It uses advanced shape warping techniques with Delaunay triangulation to create smooth, natural transitions between images.

## Features

- Automatically detects and matches feature points between images
- Uses mesh-based warping (Delaunay triangulation) for natural morphing
- Supports multiple images in sequence
- Adds audio to the final video
- Automatic generation of output file in Downloads folder

## Requirements

- Python 3.6+
- Required packages:
  - numpy
  - PIL (Pillow)
  - OpenCV (cv2)
  - scipy
  - moviepy
  - pathlib
  - wave

## Installation

```bash
pip install numpy pillow opencv-python scipy moviepy
```

## Usage

1. Place your images (.jpg, .jpeg, .png, etc.) in your Downloads folder
2. Place a WAV audio file in your Downloads folder
3. Run the program(include path to program location on the computer):
   ```bash
   python morph_video.py
   ```
4. Follow the prompts to select images, audio file, and set FPS
5. The program will automatically generate a morphed video with the selected audio
6. The output video will be saved in your Downloads folder

## How It Works

The morphing process:
1. Automatically detects feature points in each image
2. Creates a triangular mesh using Delaunay triangulation
3. Smoothly transforms triangles from one image to the next
4. Combines the transformation with cross-dissolve for smoother transitions
5. Creates a video with the specified frame rate
6. Adds the selected audio track

## Debug/Development

VS Code users can use the included launch configurations to run and debug the program.

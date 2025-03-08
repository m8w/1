MULTI-MORPHING VIDEO GENERATOR

This desktop application creates morphing videos between multiple images with synchronized audio. It combines several morphing techniques with customizable weights to create smooth, natural transitions between images.

FEATURES

- Multiple morphing techniques:
  - Delaunay triangulation for feature-based morphing
  - Cross-dissolve for simple blending transitions
  - Optical flow for motion-based morphing
  - Grid warp for creative distortion effects
  - Frequency domain morphing for unique transitions
- Customizable technique weighting system
- Support for multiple images in sequence
- Optional audio synchronization
- Adjustable FPS and transition duration
- Real-time preview during processing

REQUIREMENTS

- Python 3.6+
- Required packages:
  - numpy
  - PIL (Pillow)
  - OpenCV (cv2)
  - scipy
  - moviepy
  - pathlib
  - wave

INSTALLATION

pip install numpy pillow opencv-python scipy moviepy

USAGE

1. Run the program:
   python multi_morph_video.py

2. Use the "Select Images" button to choose your source images (.jpg, .jpeg, .png, etc.)
   - Images will be processed in the order they are selected
   - You can use "Clear Images" to start over

3. (Optional) Select an audio file to add to your video

4. Configure your morphing settings:
   - Check the techniques you want to use
   - Adjust the weight sliders for each technique (should sum to 100%)
   - Use the "Normalize Weights" button to automatically balance weights
   - Set the desired FPS and transition duration

5. Click the "Create Morphing Video" button at the bottom of the window
   - Note: You may need to resize or maximize the window if this button is not visible

6. The program will process your images and generate a video
   - A preview will update during processing
   - The final video will be saved in your specified output location

HOW IT WORKS

The morphing process:
1. Images are loaded and resized to a standard size
2. Each morphing technique processes the images differently:
   - Delaunay Triangulation: Detects facial/feature points and creates triangular meshes that transform between images
   - Cross-Dissolve: Simple alpha blending between images
   - Optical Flow: Estimates pixel movement between images for motion-based morphing
   - Grid Warp: Creates a grid and applies random offsets for artistic warping effects
   - Frequency Domain: Morphs images in the frequency domain using Fourier transforms
3. The outputs from each technique are blended according to the user-defined weights
4. Frames are assembled into a video with the specified frame rate
5. If selected, audio is added to the final video

TROUBLESHOOTING

- Window Size Issues: If you can't see all UI elements, try maximizing or resizing the application window
- Weight Normalization: If your technique weights don't sum to 100%, use the "Normalize Weights" button
- Processing Time: Complex morphs with many images may take time to process, especially at high FPS

COMING SOON

Planned features for future releases:
- Command line interface for batch processing
- Additional morphing techniques
- Custom keypoint selection for Delaunay triangulation
- Export options for different video formats and resolutions
- Progress bar with estimated time remaining
- Save/load configuration presets

DEVELOPMENT

VS Code users can use the included launch configurations to run and debug the program.


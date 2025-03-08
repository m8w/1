# Multi Morph Video Generator

![Multi Morph Video Generator Banner](path/to/banner-image.png)

## Description

Multi Morph Video Generator is a powerful Python application that creates smooth morphing transitions between multiple images. The program combines various morphing techniques with customizable weights to produce unique and visually appealing video sequences. Whether you're creating artistic videos, presentations, or visual effects, this tool provides an intuitive interface for generating high-quality image morphing animations.

## Features

- **Multiple Morphing Techniques**:
  - Delaunay Triangulation: Uses facial landmarks or feature points for natural transitions
  - Cross-Dissolve: Simple alpha blending between images
  - Optical Flow: Estimates motion between images for smooth transitions
  - Grid Warp: Creates a grid and applies controlled warping effects
  - Frequency Domain Morphing: Transforms and morphs images in the frequency domain

- **Customizable Settings**:
  - Adjustable weights for each morphing technique
  - Frame rate (FPS) control
  - Transition duration control
  - Option to add background audio

- **User-Friendly Interface**:
  - Visual preview of morphing progress
  - Simple image selection and management
  - Real-time status updates during processing

## Requirements

- Python 3.6 or higher
- Required Python packages:
  - OpenCV (cv2)
  - NumPy
  - tkinter
  - PIL (Pillow)
  - scipy
  - dlib (for facial landmark detection)
  - moviepy (for audio processing)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/multi-morph-video.git
   cd multi-morph-video
   ```

2. Install the required dependencies:
   ```
   pip install opencv-python numpy pillow scipy dlib moviepy
   ```

3. For facial landmark detection, download the shape predictor:
   ```
   wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
   ```

4. Run the application:
   ```
   python multi_morph_video.py
   ```

## Usage Guide

### 1. Image Selection

![Image Selection Interface](path/to/image-selection-screenshot.png)

- Click the "Select Images" button to choose multiple images for morphing
- Use the "Clear Images" button to remove all selected images
- Images will be processed in the order they are selected
- For best results, use images with similar dimensions and content

### 2. Configure Morphing Techniques

![Technique Configuration](path/to/technique-config-screenshot.png)

- Check the boxes for the morphing techniques you want to use
- Adjust the weight sliders to control how much each technique contributes
- Click "Normalize Weights" to automatically balance the weights to total 100%

### 3. Configure Output Settings

![Output Settings](path/to/output-settings-screenshot.png)

- Set the desired frames per second (FPS) for the output video
- Adjust the duration of each transition in seconds
- Optionally select a background audio file
- Choose an output filename and location

### 4. Generate the Video

- Ensure your window is large enough to see all UI elements
- Click the "Create Morphing Video" button at the bottom of the interface
- A progress indicator will show the current status of the morphing process
- Preview frames will be displayed as the video is generated

## Morphing Techniques Explained

### Delaunay Triangulation
This technique identifies key points in each image (like facial features) and creates triangular meshes. The images are warped by transforming these triangles from one shape to another, resulting in natural-looking morphs, especially for faces.

### Cross-Dissolve
The simplest morphing technique, cross-dissolve performs a straight alpha blend between two images. While basic, it's effective for similar images and provides a good foundation for other techniques.

### Optical Flow
Optical flow analyzes how pixels move between images and creates a flow field. This technique excels at preserving motion and is particularly effective for objects that change position between images.

### Grid Warp
Grid warping divides images into a regular grid and transforms the grid points. This creates controlled distortion effects and works well for abstract morphing effects.

### Frequency Domain Morphing
This advanced technique converts images to the frequency domain using Fourier transforms, morphs them in this domain, and converts back. It's excellent for texture transitions and can create unique effects not possible with spatial techniques.

## Troubleshooting

### Common Issues

1. **Program crashes when selecting images**
   - Ensure images are in common formats (JPG, PNG)
   - Try using smaller image files (under 10MB)
   - Check that you have sufficient memory available

2. **Video generation is extremely slow**
   - Reduce the number of selected techniques
   - Lower the FPS or transition duration
   - Use smaller input images
   - Close other resource-intensive applications

3. **"Create Morphing Video" button not visible**
   - Maximize or resize the application window
   - All UI elements may not be visible with smaller window sizes

4. **Poor quality morphing results**
   - Try adjusting technique weights
   - Use images with similar content and composition
   - Increase the weight of Delaunay triangulation for faces
   - For abstract images, increase optical flow or grid warp weights

### Error Messages

- **"Facial landmarks not detected"**: The program couldn't find faces in your images. Try using more front-facing portraits or adjust technique weights to rely less on Delaunay triangulation.

- **"Memory error during processing"**: You may be using too large images or too many images. Try reducing image size or processing fewer images at once.

## Advanced Usage

### Command Line Options

The program also supports command line operation:

```
python multi_morph_video.py --images img1.jpg img2.jpg img3.jpg --techniques delaunay=30 crossdissolve=20 opticalflow=50 --fps 30 --duration 2 --output morphed_video.mp4
```

### Custom Feature Points

Advanced users can define custom feature points for the Delaunay triangulation:

1. Create a JSON file with feature point coordinates for each image
2. Use the `--custom-points` flag when running from command line
3. Or load points through the "Load Custom Points" option in the UI

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV community for computer vision algorithms
- dlib developers for facial landmark detection
- Original image morphing research papers (referenced in code comments)

After examining the `multi_morph_video.py` script, here's how the program processes images once the settings are set:

1. The processing begins when the user clicks the "Create Morphing Video" button, which calls the `create_video()` method.

2. The key processing steps after settings are set:

   a. The program checks if at least two images are selected and if the technique weights sum to 100%.
   
   b. It loads the selected images and resizes them to a standard size (640x480).
   
   c. It sets up a video writer for the output file.
   
   d. For each pair of consecutive images:
      - It creates frames that morph between them using the selected techniques
      - The number of frames is determined by FPS × duration per transition settings
   
   e. For each frame in a transition:
      - It calculates the alpha value (0 to 1) representing the position in the transition
      - It applies each selected morphing technique with the respective alpha value
      - Each technique (Delaunay triangulation, cross-dissolve, optical flow, grid warp, frequency domain) generates its own morphed image
      - These technique outputs are blended according to the weight assigned to each technique
      - The blended frame is written to the output video
      - A preview is occasionally updated in the UI
   
   f. After all transitions are processed, it adds audio to the video if an audio file was selected.

3. The morphing techniques are applied in the following specialized methods:
   - `delaunay_morph()`: Uses facial landmarks or feature points to create triangular meshes and warps between them
   - `cross_dissolve()`: Simple alpha blending between images
   - `optical_flow_morph()`: Uses optical flow to estimate motion between images
   - `grid_warp()`: Creates a grid and warps it with random offsets
   - `frequency_domain_morph()`: Morphs images in the frequency domain using Fourier transforms

4. The final blending of technique outputs happens in the `blend_morphed_results()` method, which combines the outputs based on the assigned weights.

The program provides visual feedback during processing by updating the preview canvas and status label to show progress.

python multi_morph_video.py
^ this will start the program going,. if all the libraries are installed --
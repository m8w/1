#!/usr/bin/env python3
import os
import glob
import datetime
import numpy as np
from PIL import Image
import cv2  # OpenCV for computer vision tasks
from scipy.spatial import Delaunay  # For Delaunay triangulation
from moviepy import VideoFileClip, ImageSequenceClip, AudioFileClip
from pathlib import Path
import wave
import contextlib
import subprocess
import sys

def list_files_with_extensions(directory, extensions):
    """List all files in a directory with specified extensions."""
    files = []
    for ext in extensions:
        # Find files with the given extension
        pattern = os.path.join(directory, f"*.{ext}")
        files.extend(glob.glob(pattern))
        # Also check for uppercase extension
        pattern = os.path.join(directory, f"*.{ext.upper()}")
        files.extend(glob.glob(pattern))
    # Sort files by name
    return sorted(files)

def select_files_from_list(files, prompt_text, multiple=True):
    """Display a list of files and let the user select one or more."""
    if not files:
        print(f"No matching files found.")
        return []
    
    print(f"\n{prompt_text}")
    for i, file in enumerate(files, 1):
        filename = os.path.basename(file)
        print(f"{i}. {filename}")
    
    while True:
        try:
            if multiple:
                selection = input("\nEnter numbers separated by commas (e.g., 1,3,5) or 'a' for all: ")
                if selection.lower() == 'a':
                    return files
                if not selection.strip():
                    continue
                
                indices = [int(idx.strip()) - 1 for idx in selection.split(',') if idx.strip()]
                if not indices:
                    print("Please select at least one file.")
                    continue
                
                if any(idx < 0 or idx >= len(files) for idx in indices):
                    print(f"Invalid selection. Please enter numbers between 1 and {len(files)}.")
                    continue
                
                selected_files = [files[idx] for idx in indices]
                if len(selected_files) < 2:
                    print("Please select at least 2 images for morphing.")
                    continue
                return selected_files
            else:
                selection = input("\nEnter a number: ")
                if not selection.strip():
                    continue
                
                idx = int(selection) - 1
                if idx < 0 or idx >= len(files):
                    print(f"Invalid selection. Please enter a number between 1 and {len(files)}.")
                    continue
                
                return files[idx]
        except ValueError:
            print("Please enter valid numbers.")

def generate_output_filename():
    """Generate a timestamped filename for the output video."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    downloads_dir = str(Path.home() / "Downloads")
    return os.path.join(downloads_dir, f"morphed_video_{timestamp}.mp4")

def get_audio_duration(audio_file):
    """Get the duration of a WAV audio file in seconds."""
    with contextlib.closing(wave.open(audio_file, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)

def detect_feature_points(img, max_points=50):
    """
    Automatically detect feature points in an image using OpenCV's GoodFeaturesToTrack.
    
    Args:
        img: Input image (PIL Image or numpy array)
        max_points: Maximum number of feature points to detect
        
    Returns:
        Array of points (x, y) coordinates
    """
    # Convert to numpy array if it's a PIL Image
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    # Convert to grayscale if it's a color image
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # Detect corners/features
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_points,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=7
    )
    
    # Add border points to ensure the entire image is covered
    h, w = gray.shape
    border_points = np.array([
        [[0, 0]],
        [[w-1, 0]],
        [[0, h-1]],
        [[w-1, h-1]],
        [[w//2, 0]],
        [[w//2, h-1]],
        [[0, h//2]],
        [[w-1, h//2]]
    ], dtype=np.float32)
    
    # Combine detected corners with border points
    if corners is not None:
        points = np.vstack((corners, border_points))
    else:
        points = border_points
        
    return points.reshape(-1, 2)

def apply_affine_transform(src, src_tri, dst_tri, size):
    """
    Apply affine transform calculated using srcTri and dstTri to src and
    output an image of size.
    """
    # Given a pair of triangles, find the affine transform.
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None,
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    
    return dst

def warp_triangle(img1, img2, t1, t2, t, alpha):
    """
    Warps and alpha blends triangular regions from img1 and img2 to img
    """
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))
    
    # Offset points by left top corner of the respective rectangles
    t1_rect = []
    t2_rect = []
    t_rect = []
    
    for i in range(0, 3):
        t_rect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
    
    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_rect), (1.0, 1.0, 1.0), 16, 0)
    
    # Apply warpImage to small rectangular patches
    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2_rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    
    size = (r[2], r[3])
    warp_img1 = apply_affine_transform(img1_rect, t1_rect, t_rect, size)
    warp_img2 = apply_affine_transform(img2_rect, t2_rect, t_rect, size)
    
    # Alpha blend rectangular patches
    img_rect = (1.0 - alpha) * warp_img1 + alpha * warp_img2
    
    # Copy triangular region of the rectangular patch to the output image
    img = np.zeros_like(img1)
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * (1 - mask) + img_rect * mask
    
    return img

def morph_images(img1, img2, alpha):
    """
    Create a morphed image between img1 and img2 based on alpha (0-1) using Delaunay triangulation.
    
    Args:
        img1, img2: Input images (PIL Images)
        alpha: Morphing ratio (0 = img1, 1 = img2)
        
    Returns:
        Morphed image as PIL Image
    """
    # Ensure images are the same size and convert to numpy arrays
    img1 = np.array(img1)
    img2 = np.array(img2)
    
    # Get image dimensions
    h, w = img1.shape[:2]
    
    # Detect feature points in both images
    points1 = detect_feature_points(img1)
    points2 = detect_feature_points(img2)
    
    # Ensure both point sets have the same number of points
    min_points = min(len(points1), len(points2))
    points1 = points1[:min_points]
    points2 = points2[:min_points]
    
    # Compute weighted average point coordinates
    points = (1 - alpha) * points1 + alpha * points2
    
    # Compute Delaunay triangulation
    tri = Delaunay(points)
    simplices = tri.simplices
    
    # Initialize output image
    morphed_img = np.zeros_like(img1, dtype=np.float32)
    
    # For each triangle
    for simplex in simplices:
        # Get triangle vertices
        t1 = points1[simplex].astype(np.float32)
        t2 = points2[simplex].astype(np.float32)
        t = points[simplex].astype(np.float32)
        
        # Warp and blend triangle regions
        triangle_morphed = warp_triangle(img1, img2, t1, t2, t, alpha)
        
        # Add to output image
        morphed_img += triangle_morphed
    
    # Add cross-dissolve for smoothness
    cross_dissolve = img1 * (1 - alpha) + img2 * alpha
    # Blend the warped image with cross-dissolve for smoother transitions
    final_morphed = 0.7 * morphed_img + 0.3 * cross_dissolve
    
    return Image.fromarray(np.uint8(final_morphed))

def create_morphing_video(image_paths, audio_path, output_path, fps=30):
    """Create a video that morphs between the images with the specified audio."""
    try:
        print("\nThis morphing uses advanced shape warping with Delaunay triangulation:")
        print("  - Automatically detects feature points in each image")
        print("  - Creates a triangular mesh based on these points")
        print("  - Smoothly warps and blends these triangles for natural transformations")
        
        # Load images and ensure they're all the same size
        images = []
        target_size = None
        
        print("\nProcessing images...")
        for i, path in enumerate(image_paths):
            print(f"  Loading image {i+1}/{len(image_paths)}: {os.path.basename(path)}")
            img = Image.open(path).convert('RGB')
            
            # Set target size based on first image
            if target_size is None:
                target_size = img.size
            else:
                img = img.resize(target_size)
                
            images.append(img)
        
        # Get audio duration
        print(f"\nAnalyzing audio: {os.path.basename(audio_path)}")
        audio_duration = get_audio_duration(audio_path)
        print(f"  Audio duration: {audio_duration:.2f} seconds")
        
        # Calculate total number of frames
        total_frames = int(audio_duration * fps)
        print(f"  Total frames to generate: {total_frames} (at {fps} FPS)")
        
        # Create temporary directory for frames
        temp_dir = os.path.join(os.path.dirname(output_path), "temp_frames")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Generate frames
            print("\nGenerating video frames...")
            num_segments = len(images) - 1
            frames_per_segment = total_frames / num_segments
            
            frame_count = 0
            for i in range(num_segments):
                segment_frames = int(frames_per_segment) if i < num_segments - 1 else (total_frames - frame_count)
                
                for j in range(segment_frames):
                    alpha = j / segment_frames
                    morphed = morph_images(images[i], images[i+1], alpha)
                    
                    # Save frame
                    frame_path = os.path.join(temp_dir, f"frame_{frame_count:06d}.jpg")
                    morphed.save(frame_path, quality=95)
                    
                    frame_count += 1
                    if frame_count % 10 == 0 or frame_count == total_frames:
                        progress = (frame_count / total_frames) * 100
                        print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
            
            # Create video from frames with audio
            print(f"\nCreating final video with audio...")
            
            # Create video clip from frames
            clip = ImageSequenceClip(temp_dir, fps=fps)
            
            # Add audio
            audio_clip = AudioFileClip(audio_path)
            final_clip = clip.with_audio(audio_clip)
            
            # Write video file
            final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
            
            print(f"\nVideo successfully created: {output_path}")
            
        finally:
            # Clean up temp files
            import shutil
            if os.path.exists(temp_dir):
                print("\nCleaning up temporary files...")
                shutil.rmtree(temp_dir)
                
    except Exception as e:
        print(f"\nError creating video: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

def main():
    try:
        # Set up paths
        downloads_dir = str(Path.home() / "Downloads")
        
        # Print header
        print("\n" + "="*60)
        print("                MORPHING VIDEO GENERATOR")
        print("="*60)
        print("\nThis script will create a video that morphs between images with audio.")
        print("All files will be automatically loaded from your Downloads folder.")
        
        # List image files
        image_extensions = ["jpg", "jpeg", "png", "gif", "bmp", "tiff"]
        image_files = list_files_with_extensions(downloads_dir, image_extensions)
        
        if not image_files:
            print("\nNo image files found in Downloads folder.")
            return
            
        # Let user select images
        print(f"\nFound {len(image_files)} images in Downloads folder.")
        selected_images = select_files_from_list(
            image_files, 
            "Select at least 2 images to morph between (at least 2 required):",
            multiple=True
        )
        
        if len(selected_images) < 2:
            print("\nAt least 2 images are required for morphing. Exiting.")
            return
            
        # List audio files
        audio_extensions = ["wav"]
        audio_files = list_files_with_extensions(downloads_dir, audio_extensions)
        
        if not audio_files:
            print("\nNo WAV audio files found in Downloads folder.")
            return
            
        # Let user select audio file
        print(f"\nFound {len(audio_files)} WAV files in Downloads folder.")
        selected_audio = select_files_from_list(
            audio_files,
            "Select an audio file:",
            multiple=False
        )
        
        # Generate output filename
        output_path = generate_output_filename()
        
        # Get FPS
        while True:
            fps_input = input("\nEnter frames per second (FPS) [default=30]: ")
            if not fps_input:
                fps = 30
                break
                
            try:
                fps = int(fps_input)
                if fps < 1:
                    print("FPS must be at least 1.")
                    continue
                break
            except ValueError:
                print("Please enter a valid number.")
        
        # Show summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Selected images ({len(selected_images)}):")
        for i, img in enumerate(selected_images, 1):
            print(f"  {i}. {os.path.basename(img)}")
            
        print(f"\nSelected audio: {os.path.basename(selected_audio)}")
        print(f"Output video: {os.path.basename(output_path)}")
        print(f"FPS: {fps}")
        
        # Confirm and create video
        confirm = input("\nProceed with video creation? (y/n) [y]: ")
        if confirm.lower() not in ['', 'y', 'yes']:
            print("Operation cancelled. Exiting.")
            return
            
        # Create the video
        success = create_morphing_video(selected_images, selected_audio, output_path, fps)
        
        if success:
            print("\nVideo creation completed successfully!")
            
            # Try to open the folder containing the video
            try:
                if sys.platform == 'darwin':  # macOS
                    subprocess.run(['open', os.path.dirname(output_path)])
                elif sys.platform == 'win32':  # Windows
                    subprocess.run(['explorer', os.path.dirname(output_path)])
                elif sys.platform.startswith('linux'):  # Linux
                    subprocess.run(['xdg-open', os.path.dirname(output_path)])
            except:
                pass
        else:
            print("\nVideo creation failed.")
            
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user. Exiting.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


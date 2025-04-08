#!/usr/bin/env python3
import os
import glob
import datetime
import numpy as np
from PIL import Image
import cv2
from scipy.spatial import Delaunay
import tempfile
import shutil
import subprocess
import sys
import argparse
from pathlib import Path
import contextlib
import time
import math
from tqdm import tqdm

# Constants
SSD_DIR = "/Volumes/ssd1"
TEMP_DIR = os.path.join(SSD_DIR, "morph_data", "temp_video_frames")

def check_ssd_available():
    """Check if the SSD is mounted and writable."""
    # Check if SSD is mounted
    if not os.path.exists(SSD_DIR):
        print(f"Error: SSD not found at {SSD_DIR}")
        return False
        
    # Check if SSD is writable
    try:
        test_file = os.path.join(SSD_DIR, ".write_test")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        
        # Create morph_data directory if it doesn't exist
        morph_data_dir = os.path.join(SSD_DIR, "morph_data")
        os.makedirs(morph_data_dir, exist_ok=True)
        
        return True
    except Exception as e:
        print(f"Error: SSD at {SSD_DIR} is not writable: {e}")
        return False

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

def generate_output_filename(suffix=""):
    """Generate a timestamped filename for the output video."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(SSD_DIR, f"morphed_video_{timestamp}{suffix}.mp4")

# ---- Image Morphing Functions ----

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
        img1, img2: Input images (PIL Images or numpy arrays)
        alpha: Morphing ratio (0 = img1, 1 = img2)
        
    Returns:
        Morphed image as PIL Image
    """
    # Ensure images are numpy arrays
    if isinstance(img1, Image.Image):
        img1 = np.array(img1)
    if isinstance(img2, Image.Image):
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
    
    # Return the final morphed image (as numpy array for efficiency when processing videos)
    return np.uint8(final_morphed)

# ---- Video Processing Functions ----

def get_video_info(video_path):
    """Extract metadata from video file using ffprobe."""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,nb_frames',
        '-of', 'csv=p=0',
        video_path
    ]
    
    try:
        output = subprocess.check_output(cmd).decode('utf-8').strip().split(',')
        width, height, frame_rate, nb_frames = output
        
        # Parse frame rate fraction (e.g., "30000/1001")
        if '/' in frame_rate:
            num, den = map(int, frame_rate.split('/'))
            fps = num / den
        else:
            fps = float(frame_rate)
            
        return {
            'width': int(width),
            'height': int(height),
            'fps': fps,
            'frame_count': int(nb_frames) if nb_frames != 'N/A' else None
        }
    except Exception as e:
        print(f"Error getting video info: {e}")
        return None

def extract_frames(video_path, output_dir, keyframes_only=False, keyframe_interval=None):
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        keyframes_only: If True, only extract keyframes
        keyframe_interval: If provided, extract frames at this interval
        
    Returns:
        List of paths to extracted frames
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video info
    video_info = get_video_info(video_path)
    if not video_info:
        print("Failed to get video information")
        return []
        
    # Extract frames
    if keyframes_only and keyframe_interval:
        # Extract frames at specified interval
        frame_count = video_info['frame_count']
        if not frame_count:
            # If frame count is not available, estimate by duration * fps
            cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
                   '-of', 'csv=p=0', video_path]
            duration = float(subprocess.check_output(cmd).decode('utf-8').strip())
            frame_count = int(duration * video_info['fps'])
            
        keyframes = []
        for i in range(0, frame_count, keyframe_interval):
            output_file = os.path.join(output_dir, f"frame_{i:06d}.jpg")
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vf', f'select=eq(n\\,{i})',
                '-vframes', '1',
                '-q:v', '2',
                output_file
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(output_file):
                keyframes.append((i, output_file))
                
        return keyframes
    else:
        # Extract all frames
        # Extract all frames
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-q:v', '2',  # Quality level (lower is better)
            os.path.join(output_dir, 'frame_%06d.jpg')
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Get list of all extracted frames
        frames = sorted(glob.glob(os.path.join(output_dir, 'frame_*.jpg')))
        return [(int(os.path.basename(f).split('_')[1].split('.')[0]), f) for f in frames]

def detect_scene_changes(video_path, threshold=30, min_scene_length=24):
    """
    Detect scene changes in a video using FFmpeg's scene detection filter.
    
    Args:
        video_path: Path to the video file
        threshold: Scene change detection threshold (0-100)
        min_scene_length: Minimum number of frames between scene changes
        
    Returns:
        List of frame numbers where scene changes occur
    """
    temp_output = tempfile.NamedTemporaryFile(suffix='.txt', delete=False).name
    
    try:
        # Use FFmpeg's scene detection filter
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'select=gt(scene\\,{threshold/100}),metadata=print:file={temp_output}',
            '-f', 'null',
            '-'
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Parse the output file to get scene change frame numbers
        scene_changes = []
        prev_scene = -min_scene_length  # Initialize to allow first scene
        
        if os.path.exists(temp_output):
            with open(temp_output, 'r') as f:
                for line in f:
                    if 'pts_time' in line:
                        parts = line.strip().split('=')
                        if len(parts) >= 2:
                            time_str = parts[1].split(' ')[0]
                            try:
                                time_sec = float(time_str)
                                # Convert time to frame number
                                video_info = get_video_info(video_path)
                                if video_info:
                                    frame_num = int(time_sec * video_info['fps'])
                                    # Only add if it's sufficiently far from the previous scene change
                                    if frame_num - prev_scene >= min_scene_length:
                                        scene_changes.append(frame_num)
                                        prev_scene = frame_num
                            except ValueError:
                                pass
            
        return scene_changes
    finally:
        # Clean up temp file
        if os.path.exists(temp_output):
            os.unlink(temp_output)

def auto_select_keyframes(video_path, num_keyframes=10, use_scene_detection=True):
    """
    Automatically select keyframes from a video.
    
    Args:
        video_path: Path to the video file
        num_keyframes: Number of keyframes to select
        use_scene_detection: Whether to use scene detection
        
    Returns:
        List of frame numbers to use as keyframes
    """
    video_info = get_video_info(video_path)
    if not video_info or not video_info['frame_count']:
        print("Failed to get video frame count")
        return []
        
    frame_count = video_info['frame_count']
    
    # If using scene detection, prioritize scene changes
    if use_scene_detection:
        scene_changes = detect_scene_changes(video_path)
        
        if len(scene_changes) > 0:
            # If we have more scene changes than requested keyframes, select a subset
            if len(scene_changes) > num_keyframes:
                # Evenly distribute selected scene changes
                step = len(scene_changes) / num_keyframes
                keyframes = [scene_changes[int(i * step)] for i in range(num_keyframes)]
            else:
                keyframes = scene_changes
                
            # Ensure first and last frames are included
            if 0 not in keyframes:
                keyframes.insert(0, 0)
            if frame_count-1 not in keyframes:
                keyframes.append(frame_count-1)
                
            return sorted(keyframes)
    
    # Fall back to evenly spaced keyframes
    step = frame_count / (num_keyframes - 1)
    keyframes = [int(i * step) for i in range(num_keyframes)]
    
    # Ensure last frame is exactly the final frame
    keyframes[-1] = frame_count - 1
    
    return keyframes

def blend_frames(original, morphed, blend_factor):
    """
    Blend original and morphed frames using the specified blend factor.
    
    Args:
        original: Original frame (numpy array)
        morphed: Morphed frame (numpy array)
        blend_factor: Blend factor (0 = original, 1 = morphed)
        
    Returns:
        Blended frame
    """
    return cv2.addWeighted(original, 1.0 - blend_factor, morphed, blend_factor, 0)

def process_video_frames(video_path, keyframes, output_dir, morph_strength=0.7, 
                         transition_frames=15, blend_with_original=True):
    """
    Process video frames with morphing effects between keyframes.
    
    Args:
        video_path: Path to the video file
        keyframes: List of frame numbers to use as keyframes
        output_dir: Directory to save processed frames
        morph_strength: Strength of the morphing effect (0-1)
        transition_frames: Number of frames for each transition
        blend_with_original: Whether to blend morphed frames with original frames
        
    Returns:
        Directory containing processed frames
    """
    # Create directories
    frames_dir = os.path.join(output_dir, 'extracted_frames')
    morphed_dir = os.path.join(output_dir, 'morphed_frames')
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(morphed_dir, exist_ok=True)
    
    # Extract video info
    video_info = get_video_info(video_path)
    if not video_info:
        print("Failed to get video information")
        return None
        
    total_frames = video_info['frame_count']
    fps = video_info['fps']
    
    print(f"\nExtracting all frames from video...")
    frame_list = extract_frames(video_path, frames_dir)
    
    if not frame_list:
        print("Failed to extract frames")
        return None
        
    # Create a mapping of frame number to frame path
    frame_map = {frame_num: path for frame_num, path in frame_list}
    
    print(f"\nExtracting {len(keyframes)} keyframes...")
    keyframe_images = {}
    for kf in keyframes:
        if kf in frame_map:
            keyframe_images[kf] = cv2.imread(frame_map[kf])
    
    if len(keyframe_images) < 2:
        print("Not enough valid keyframes found")
        return None
        
    # Sort keyframes
    sorted_keyframes = sorted(keyframe_images.keys())
    
    # Process frames
    print(f"\nProcessing video with morphing effects...")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Using {len(sorted_keyframes)} keyframes")
    print(f"  - Morphing strength: {morph_strength}")
    print(f"  - Transition frames: {transition_frames}")
    
    # Create progress bar
    with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
        # Process each segment between keyframes
        for i in range(len(sorted_keyframes) - 1):
            start_kf = sorted_keyframes[i]
            end_kf = sorted_keyframes[i+1]
            segment_length = end_kf - start_kf
            
            # Skip if segment is too short
            if segment_length <= 1:
                continue
                
            start_img = keyframe_images[start_kf]
            end_img = keyframe_images[end_kf]
            
            # Determine transition regions
            half_transition = min(transition_frames // 2, segment_length // 4)
            
            # Process each frame in the segment
            for f in range(start_kf, end_kf + 1):
                # Determine if this frame needs morphing
                frame_position = f - start_kf
                segment_progress = frame_position / segment_length
                
                # Read original frame
                orig_frame = None
                if f in frame_map:
                    orig_frame = cv2.imread(frame_map[f])
                else:
                    # Skip if frame doesn't exist
                    pbar.update(1)
                    continue
                
                # Determine morphing behavior based on frame position
                if frame_position <= half_transition:
                    # Start transition: gradually increase morphing
                    transition_progress = frame_position / half_transition
                    # Morph between start keyframe and interpolated frame
                    alpha = segment_progress
                    morphed = morph_images(start_img, end_img, alpha)
                    # Blend with original based on transition progress
                    blend_factor = transition_progress * morph_strength
                elif frame_position >= (segment_length - half_transition):
                    # End transition: gradually decrease morphing
                    remaining = segment_length - frame_position
                    transition_progress = remaining / half_transition
                    # Morph between interpolated frame and end keyframe
                    alpha = segment_progress
                    morphed = morph_images(start_img, end_img, alpha)
                    # Blend with original based on transition progress
                    blend_factor = transition_progress * morph_strength
                else:
                    # Middle region: full morphing effect
                    alpha = segment_progress
                    morphed = morph_images(start_img, end_img, alpha)
                    blend_factor = morph_strength
                
                # Blend morphed frame with original if requested
                if blend_with_original and orig_frame is not None:
                    final_frame = blend_frames(orig_frame, morphed, blend_factor)
                else:
                    final_frame = morphed
                
                # Save processed frame
                output_path = os.path.join(morphed_dir, f"frame_{f:06d}.jpg")
                cv2.imwrite(output_path, final_frame)
                
                pbar.update(1)
        
    return morphed_dir

def create_output_video(frames_dir, output_path, audio_source=None, fps=30):
    """
    Create a video from processed frames.
    
    Args:
        frames_dir: Directory containing processed frames
        output_path: Path to save the output video
        audio_source: Path to audio source (usually the original video)
        fps: Frames per second
        
    Returns:
        Path to the created video
    """
    print(f"\nCreating output video...")
    
    # Get all frames
    frames = sorted(glob.glob(os.path.join(frames_dir, 'frame_*.jpg')))
    if not frames:
        print("No processed frames found")
        return None
    
    # Create temporary video without audio
    temp_video = os.path.join(os.path.dirname(output_path), "temp_video.mp4")
    
    # Use ffmpeg to create video from frames
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-framerate', str(fps),
        '-i', os.path.join(frames_dir, 'frame_%06d.jpg'),
        '-c:v', 'libx264',
        '-preset', 'slow',
        '-crf', '18',  # Quality (0-51, lower is better)
        '-pix_fmt', 'yuv420p',
        temp_video
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if not os.path.exists(temp_video):
        print("Failed to create video from frames")
        return None
    
    # Add audio if source provided
    if audio_source and os.path.exists(audio_source):
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-i', temp_video,
            '-i', audio_source,
            '-c:v', 'copy',  # Copy video stream without re-encoding
            '-c:a', 'aac',   # Re-encode audio to AAC
            '-map', '0:v:0', # Use video from first input
            '-map', '1:a:0', # Use audio from second input
            '-shortest',     # End when shortest input ends
            output_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Remove temporary video
        os.remove(temp_video)
        
        if os.path.exists(output_path):
            return output_path
        else:
            print("Failed to add audio to video")
            # Fall back to using the temp video if it exists
            if os.path.exists(temp_video):
                shutil.move(temp_video, output_path)
                return output_path
            return None
    else:
        # No audio source, just rename the temp video
        shutil.move(temp_video, output_path)
        return output_path

def clean_up_temp_files(temp_dir):
    """Remove temporary files after processing."""
    if os.path.exists(temp_dir):
        print(f"\nCleaning up temporary files...")
        try:
            shutil.rmtree(temp_dir)
            print(f"Removed temporary directory: {temp_dir}")
        except Exception as e:
            print(f"Warning: Failed to remove temporary directory: {e}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Video Morpher - Create morphing effects between video frames')
    
    parser.add_argument('-i', '--input', help='Input video file path')
    parser.add_argument('-o', '--output', help='Output video file path')
    parser.add_argument('-k', '--keyframes', type=int, default=5, 
                        help='Number of keyframes to use for automatic selection')
    parser.add_argument('-s', '--strength', type=float, default=0.7, 
                        help='Morphing effect strength (0.0-1.0)')
    parser.add_argument('-t', '--transitions', type=int, default=15, 
                        help='Number of frames for each transition')
    parser.add_argument('--scene-detection', action='store_true', 
                        help='Use scene detection for automatic keyframe selection')
    parser.add_argument('--manual-keyframes', action='store_true', 
                        help='Manually select keyframes from video')
    
    return parser.parse_args()

def main():
    """Main function for the video morpher."""
    try:
        # Print header
        print("\n" + "="*60)
        print("                  VIDEO MORPHER")
        print("="*60)
        print("\nThis tool creates morphing effects between frames in a video.")

        # Check if SSD is available
        if not check_ssd_available():
            print("Cannot proceed: SSD is not available or not writable.")
            return

        # Parse command line arguments
        args = parse_arguments()
        
        # Create temporary directory
        temp_dir = os.path.join(TEMP_DIR, f"morph_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Select input video file
            input_video = args.input
            if not input_video or not os.path.exists(input_video):
                # List video files
                video_extensions = ["mp4", "avi", "mov", "mkv", "webm"]
                video_files = list_files_with_extensions(SSD_DIR, video_extensions)
                
                if not video_files:
                    print("\nNo video files found in SSD folder.")
                    return
                
                print(f"\nFound {len(video_files)} video files in SSD folder.")
                input_video = select_files_from_list(
                    video_files,
                    "Select a video file to process:",
                    multiple=False
                )
                
                if not input_video:
                    print("No video selected. Exiting.")
                    return
            
            # Generate output filename if not provided
            output_video = args.output
            if not output_video:
                output_video = generate_output_filename()
            
            # Get video information
            video_info = get_video_info(input_video)
            if not video_info:
                print("Failed to read video information. Exiting.")
                return
                
            print(f"\nVideo information:")
            print(f"  - Resolution: {video_info['width']}x{video_info['height']}")
            print(f"  - FPS: {video_info['fps']}")
            print(f"  - Total frames: {video_info['frame_count']}")
            
            # Keyframe selection method
            keyframes = []
            if args.manual_keyframes:
                print("\nManual keyframe selection:")
                # Extract frames at intervals for preview
                preview_dir = os.path.join(temp_dir, "preview_frames")
                os.makedirs(preview_dir, exist_ok=True)
                
                # Calculate interval to extract a reasonable number of preview frames
                frame_count = video_info['frame_count']
                interval = max(1, frame_count // 100)  # About 100 preview frames
                
                # Extract preview frames
                preview_frames = extract_frames(input_video, preview_dir, 
                                              keyframes_only=True, 
                                              keyframe_interval=interval)
                
                if not preview_frames:
                    print("Failed to extract preview frames. Using automatic selection.")
                    keyframes = auto_select_keyframes(
                        input_video, 
                        num_keyframes=args.keyframes,
                        use_scene_detection=args.scene_detection
                    )
                else:
                    # Let user select keyframes from the preview
                    print(f"\nExtracted {len(preview_frames)} preview frames.")
                    frame_files = [path for _, path in preview_frames]
                    selected_frames = select_files_from_list(
                        frame_files,
                        "Select at least 2 keyframes for morphing:",
                        multiple=True
                    )
                    
                    if len(selected_frames) < 2:
                        print("At least 2 keyframes are required. Using automatic selection.")
                        keyframes = auto_select_keyframes(
                            input_video,
                            num_keyframes=args.keyframes,
                            use_scene_detection=args.scene_detection
                        )
                    else:
                        # Convert selected frames back to frame numbers
                        keyframes = []
                        for path in selected_frames:
                            frame_num = int(os.path.basename(path).split('_')[1].split('.')[0])
                            keyframes.append(frame_num)
                        keyframes.sort()
            else:
                # Automatic keyframe selection
                print("\nUsing automatic keyframe selection...")
                keyframes = auto_select_keyframes(
                    input_video,
                    num_keyframes=args.keyframes,
                    use_scene_detection=args.scene_detection
                )
                
            if not keyframes or len(keyframes) < 2:
                print("Failed to select keyframes. Exiting.")
                return
                
            print(f"\nSelected {len(keyframes)} keyframes:")
            print(f"  Frame numbers: {keyframes}")
            
            # Configure morphing parameters
            morph_strength = args.strength
            transition_frames = args.transitions
            
            # Process video frames
            print("\n" + "="*60)
            print("              PROCESSING VIDEO")
            print("="*60)
            
            # Process frames
            processed_dir = process_video_frames(
                input_video,
                keyframes,
                temp_dir,
                morph_strength=morph_strength,
                transition_frames=transition_frames,
                blend_with_original=True
            )
            
            if not processed_dir:
                print("Failed to process video frames. Exiting.")
                return
                
            # Create output video
            print("\n" + "="*60)
            print("           CREATING OUTPUT VIDEO")
            print("="*60)
            
            final_video = create_output_video(
                processed_dir,
                output_video,
                audio_source=input_video,
                fps=video_info['fps']
            )
            
            if final_video:
                print("\n" + "="*60)
                print("                 SUCCESS!")
                print("="*60)
                print(f"\nVideo with morphing effects created successfully:")
                print(f"  {final_video}")
                
                # Try to open the folder containing the video
                try:
                    if sys.platform == 'darwin':  # macOS
                        subprocess.run(['open', os.path.dirname(output_video)])
                    elif sys.platform == 'win32':  # Windows
                        subprocess.run(['explorer', os.path.dirname(output_video)])
                    elif sys.platform.startswith('linux'):  # Linux
                        subprocess.run(['xdg-open', os.path.dirname(output_video)])
                except:
                    pass
            else:
                print("\nFailed to create output video.")
                
        finally:
            # Clean up temporary files
            clean_up_temp_files(temp_dir)
            
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user. Exiting.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

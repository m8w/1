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
import json
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Constants
SSD_DIR = "/Volumes/ssd1"
MORPH_DATA_DIR = os.path.join(SSD_DIR, "morph_data")
TEMP_DIR = os.path.join(MORPH_DATA_DIR, "temp_video_frames")
CONFIG_FILE = os.path.join(MORPH_DATA_DIR, "batch_morph_config.json")
LOG_DIR = os.path.join(MORPH_DATA_DIR, "logs")
VERSION = "1.1.0"  # Batch version
# Check if the required directories are available
def check_ssd_available():
    """
    Check if the SSD is mounted and writable.
    
    Returns:
        bool: True if SSD is available and writable, False otherwise
    """
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
        os.makedirs(MORPH_DATA_DIR, exist_ok=True)
        os.makedirs(TEMP_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        
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

# ---- Batch Processing Functions ----

def setup_logging(video_name):
    """Set up logging for a specific video processing job."""
    # Create a unique log file name based on the input video name
    base_name = os.path.splitext(os.path.basename(video_name))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"morph_{base_name}_{timestamp}.log")
    
    # Configure logging
    logger = logging.getLogger(base_name)
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    
    return logger

def save_config(config):
    """Save configuration to a JSON file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        print(f"Warning: Failed to save configuration: {e}")
        return False

def load_config():
    """Load configuration from JSON file or return default configuration."""
    default_config = {
        "last_used": {
            "keyframes": 5,
            "strength": 0.7,
            "transitions": 15,
            "scene_detection": True,
            "manual_keyframes": False
        },
        "saved_presets": {
            "default": {
                "keyframes": 5,
                "strength": 0.7,
                "transitions": 15,
                "scene_detection": True,
                "manual_keyframes": False
            },
            "subtle": {
                "keyframes": 3,
                "strength": 0.4,
                "transitions": 30,
                "scene_detection": True,
                "manual_keyframes": False
            },
            "intense": {
                "keyframes": 8,
                "strength": 0.9,
                "transitions": 10,
                "scene_detection": True,
                "manual_keyframes": False
            }
        },
        "batch_processing": {
            "max_workers": 1,  # Default to single-process for stability
            "delete_temp_files": True,
            "auto_open_result": False
        }
    }
    
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            # Merge with default config to ensure all keys exist
            # Update nested dictionaries
            if "last_used" not in config:
                config["last_used"] = default_config["last_used"]
            if "saved_presets" not in config:
                config["saved_presets"] = default_config["saved_presets"]
            if "batch_processing" not in config:
                config["batch_processing"] = default_config["batch_processing"]
            return config
        else:
            # No config file exists, create it with default values
            save_config(default_config)
            return default_config
    except Exception as e:
        print(f"Warning: Failed to load configuration: {e}")
        return default_config

def update_last_used_config(args):
    """Update the last used configuration based on successful processing."""
    config = load_config()
    
    # Update last used settings
    config["last_used"]["keyframes"] = args.keyframes
    config["last_used"]["strength"] = args.strength
    config["last_used"]["transitions"] = args.transitions
    config["last_used"]["scene_detection"] = args.scene_detection
    config["last_used"]["manual_keyframes"] = args.manual_keyframes
    
    save_config(config)

def save_preset(name, args):
    """Save current settings as a named preset."""
    config = load_config()
    
    # Create or update the preset
    config["saved_presets"][name] = {
        "keyframes": args.keyframes,
        "strength": args.strength,
        "transitions": args.transitions,
        "scene_detection": args.scene_detection,
        "manual_keyframes": args.manual_keyframes
    }
    
    save_config(config)
    print(f"Preset '{name}' saved successfully.")

def list_presets():
    """List all available presets."""
    config = load_config()
    presets = config.get("saved_presets", {})
    
    if not presets:
        print("No presets found.")
        return
    
    print("\nAvailable presets:")
    for name, settings in presets.items():
        print(f"  {name}:")
        print(f"    Keyframes: {settings['keyframes']}")
        print(f"    Strength: {settings['strength']}")
        print(f"    Transitions: {settings['transitions']}")
        print(f"    Scene Detection: {settings['scene_detection']}")
        print(f"    Manual Keyframes: {settings['manual_keyframes']}")
        print()

def apply_preset(args, preset_name):
    """Apply a named preset to the current arguments."""
    config = load_config()
    presets = config.get("saved_presets", {})
    
    if preset_name not in presets:
        print(f"Preset '{preset_name}' not found.")
        return args
    
    preset = presets[preset_name]
    args.keyframes = preset["keyframes"]
    args.strength = preset["strength"]
    args.transitions = preset["transitions"]
    args.scene_detection = preset["scene_detection"]
    args.manual_keyframes = preset["manual_keyframes"]
    
    print(f"Applied preset '{preset_name}'")
    return args

def process_single_video(input_video, output_video=None, args=None, logger=None):
    """
    Process a single video with morphing effects.
    
    Args:
        input_video: Path to input video
        output_video: Path to output video (or None to auto-generate)
        args: Processing arguments
        logger: Logger for this specific video
        
    Returns:
        Path to output video if successful, None otherwise
    """
    if logger is None:
        logger = logging.getLogger()
    
    # Create a timestamp for this processing job
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = f"{os.path.splitext(os.path.basename(input_video))[0]}_{timestamp}"
    
    # Create temporary directory for this job
    temp_dir = os.path.join(TEMP_DIR, f"batch_{job_id}")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        logger.info(f"Starting processing of video: {input_video}")
        print(f"\nProcessing video: {os.path.basename(input_video)}")
        
        # Generate output filename if not provided
        if not output_video:
            output_video = generate_output_filename(f"_{os.path.splitext(os.path.basename(input_video))[0]}")
            logger.info(f"Generated output filename: {output_video}")
        
        # Get video information
        video_info = get_video_info(input_video)
        if not video_info:
            logger.error("Failed to read video information. Exiting.")
            print("  Failed to read video information.")
            return None
            
        logger.info(f"Video info: {video_info['width']}x{video_info['height']} @ {video_info['fps']}fps, {video_info['frame_count']} frames")
        
        # Keyframe selection method
        keyframes = []
        if args.manual_keyframes:
            logger.info("Manual keyframe selection requested but not supported in batch mode. Using automatic selection.")
            print("  Using automatic keyframe selection (manual not supported in batch mode)")
            keyframes = auto_select_keyframes(
                input_video, 
                num_keyframes=args.keyframes,
                use_scene_detection=args.scene_detection
            )
        else:
            # Automatic keyframe selection
            logger.info(f"Using automatic keyframe selection with {args.keyframes} keyframes")
            print(f"  Selecting {args.keyframes} keyframes automatically" + 
                  (" with scene detection" if args.scene_detection else ""))
            keyframes = auto_select_keyframes(
                input_video,
                num_keyframes=args.keyframes,
                use_scene_detection=args.scene_detection
            )
            
        if not keyframes or len(keyframes) < 2:
            logger.error("Failed to select keyframes. Exiting.")
            print("  Failed to select keyframes.")
            return None
            
        logger.info(f"Selected {len(keyframes)} keyframes: {keyframes}")
        
        # Process frames
        logger.info(f"Processing video with morphing strength {args.strength} and transition frames {args.transitions}")
        print(f"  Processing with strength={args.strength}, transitions={args.transitions}")
        
        processed_dir = process_video_frames(
            input_video,
            keyframes,
            temp_dir,
            morph_strength=args.strength,
            transition_frames=args.transitions,
            blend_with_original=True
        )
        
        if not processed_dir:
            logger.error("Failed to process video frames.")
            print("  Failed to process video frames.")
            return None
            
        # Create output video
        logger.info("Creating output video...")
        print("  Creating output video...")
        
        final_video = create_output_video(
            processed_dir,
            output_video,
            audio_source=input_video,
            fps=video_info['fps']
        )
        
        if final_video:
            logger.info(f"Video processing successful: {final_video}")
            print(f"  ✅ Success! Output saved to: {os.path.basename(final_video)}")
            return final_video
        else:
            logger.error("Failed to create output video.")
            print("  Failed to create output video.")
            return None
            
    except Exception as e:
        logger.exception(f"An error occurred during processing: {e}")
        print(f"  Error: {e}")
        return None
    finally:
        # Clean up temporary files if configured to do so
        config = load_config()
        if config["batch_processing"]["delete_temp_files"]:
            clean_up_temp_files(temp_dir)
            logger.info("Temporary files cleaned up")
        else:
            logger.info(f"Temporary files retained at: {temp_dir}")

def process_multiple_videos(video_paths, args, output_dir=None):
    """
    Process multiple videos with the same settings.
    
    This is the main batch processing function that handles:
    - Processing multiple videos with the same settings
    - Parallel processing using multiple workers when configured
    - Progress tracking and reporting
    - Error handling for individual video processing failures
    
    Args:
        video_paths: List of paths to input videos
        args: Processing arguments
        output_dir: Directory to save output videos (or None to use SSD root)
        
    Returns:
        List of paths to successfully processed videos
    """
    if not video_paths:
        print("No videos provided for batch processing.")
        return []
    
    print(f"\n{'='*60}")
    print(f"       BATCH PROCESSING {len(video_paths)} VIDEOS")
    print(f"{'='*60}")
    
    # Setup batch processing parameters
    config = load_config()
    max_workers = config["batch_processing"]["max_workers"]
    
    # Initialize results list
    successful_videos = []
    failed_videos = []
    
    # Use parallel processing if configured for multiple workers
    if max_workers > 1 and len(video_paths) > 1:
        print(f"\nUsing {max_workers} parallel processes for batch processing")
        
        # Process videos in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all video processing jobs
            future_to_video = {}
            for video_path in video_paths:
                # Setup logging for this video
                logger = setup_logging(video_path)
                
                # Generate output path
                if output_dir:
                    output_name = f"morphed_{os.path.splitext(os.path.basename(video_path))[0]}.mp4"
                    output_path = os.path.join(output_dir, output_name)
                else:
                    output_path = None  # Will be auto-generated
                
                # Submit the job
                future = executor.submit(
                    process_single_video, 
                    video_path, 
                    output_path, 
                    args,
                    logger
                )
                future_to_video[future] = video_path
            
            # Process results as they complete
            with tqdm(total=len(video_paths), desc="Processing videos", unit="video") as pbar:
                for future in as_completed(future_to_video):
                    video_path = future_to_video[future]
                    try:
                        output_path = future.result()
                        if output_path:
                            successful_videos.append(output_path)
                        else:
                            failed_videos.append(video_path)
                    except Exception as e:
                        print(f"\nError processing {os.path.basename(video_path)}: {e}")
                        failed_videos.append(video_path)
                    finally:
                        pbar.update(1)
    else:
        # Process videos sequentially
        with tqdm(total=len(video_paths), desc="Processing videos", unit="video") as pbar:
            for video_path in video_paths:
                # Setup logging for this video
                logger = setup_logging(video_path)
                
                # Generate output path
                if output_dir:
                    output_name = f"morphed_{os.path.splitext(os.path.basename(video_path))[0]}.mp4"
                    output_path = os.path.join(output_dir, output_name)
                else:
                    output_path = None  # Will be auto-generated
                
                # Process the video
                result = process_single_video(video_path, output_path, args, logger)
                
                if result:
                    successful_videos.append(result)
                else:
                    failed_videos.append(video_path)
                
                pbar.update(1)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"            BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total videos: {len(video_paths)}")
    print(f"Successfully processed: {len(successful_videos)}")
    print(f"Failed: {len(failed_videos)}")
    
    if successful_videos:
        print("\nSuccessfully processed videos:")
        for i, path in enumerate(successful_videos, 1):
            print(f"  {i}. {os.path.basename(path)}")
    
    if failed_videos:
        print("\nFailed videos:")
        for i, path in enumerate(failed_videos, 1):
            print(f"  {i}. {os.path.basename(path)}")
    
    # Remember successful settings for future use
    if successful_videos:
        update_last_used_config(args)
    
    return successful_videos

def parse_arguments():
    """
    Parse command line arguments.
    
    Defines and parses all the command-line options for the batch video morpher:
    - Input/output options
    - Batch processing settings
    - Morphing effect parameters
    - Preset management
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description=f'Batch Video Morpher v{VERSION} - Create morphing effects between video frames')
    
    # Input/Output options
    input_group = parser.add_argument_group('Input/Output Options')
    input_group.add_argument('-i', '--input', help='Input video file path')
    input_group.add_argument('-o', '--output', help='Output video file path')
    input_group.add_argument('--output-dir', help='Output directory for batch processing')
    
    # Batch processing options
    batch_group = parser.add_argument_group('Batch Processing Options')
    batch_group.add_argument('--batch', action='store_true', 
                         help='Enable batch processing of multiple videos')
    batch_group.add_argument('--workers', type=int, 
                         help='Number of parallel workers for batch processing')
    batch_group.add_argument('--keep-temp', action='store_true',
                         help='Keep temporary files after processing')
    
    # Morphing options
    morph_group = parser.add_argument_group('Morphing Options')
    morph_group.add_argument('-k', '--keyframes', type=int, default=5, 
                        help='Number of keyframes to use for automatic selection')
    morph_group.add_argument('-s', '--strength', type=float, default=0.7, 
                        help='Morphing effect strength (0.0-1.0)')
    morph_group.add_argument('-t', '--transitions', type=int, default=15, 
                        help='Number of frames for each transition')
    morph_group.add_argument('--scene-detection', action='store_true', 
                        help='Use scene detection for automatic keyframe selection')
    morph_group.add_argument('--manual-keyframes', action='store_true', 
                        help='Manually select keyframes from video')
    
    # Preset management
    preset_group = parser.add_argument_group('Preset Management')
    preset_group.add_argument('--preset', help='Use a saved preset')
    preset_group.add_argument('--save-preset', help='Save current settings as a preset')
    preset_group.add_argument('--list-presets', action='store_true', 
                         help='List all available presets')
    preset_group.add_argument('--last-used', action='store_true',
                         help='Use settings from last successful processing')
    
    return parser.parse_args()

def batch_main():
    """
    Main function for batch video morpher.
    
    This is the entry point for the application that handles:
    1. Command line argument parsing
    2. Preset management
    3. Batch or single video processing
    4. Error handling and cleanup
    
    All processing options are configured through command line arguments
    or saved presets.
    """
    try:
        # Print header
        print("\n" + "="*60)
        print(f"            BATCH VIDEO MORPHER v{VERSION}")
        print("="*60)
        print("\nThis tool creates morphing effects between frames in multiple videos.")

        # Check if SSD is available
        if not check_ssd_available():
            print("Cannot proceed: SSD is not available or not writable.")
            return

        # Parse command line arguments
        args = parse_arguments()
        
        # Handle preset management
        if args.list_presets:
            list_presets()
            return
            
        # Apply preset if specified
        if args.preset:
            args = apply_preset(args, args.preset)
        
        # Apply last used settings if requested
        if args.last_used:
            config = load_config()
            last_used = config.get("last_used", {})
            args.keyframes = last_used.get("keyframes", args.keyframes)
            args.strength = last_used.get("strength", args.strength)
            args.transitions = last_used.get("transitions", args.transitions)
            args.scene_detection = last_used.get("scene_detection", args.scene_detection)
            print("Using settings from last successful processing")
        
        # Save preset if requested
        if args.save_preset:
            save_preset(args.save_preset, args)
            print(f"Saved current settings as preset '{args.save_preset}'")
            if not args.batch and not args.input:
                return

        # Set up workers for batch processing
        if args.workers is not None:
            config = load_config()
            config["batch_processing"]["max_workers"] = args.workers
            save_config(config)
            print(f"Set parallel workers to {args.workers}")
        
        # Set temp file handling
        if args.keep_temp:
            config = load_config()
            config["batch_processing"]["delete_temp_files"] = False
            save_config(config)
            print("Temporary files will be kept after processing")
        
        # Create temporary directory for this session
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = os.path.join(TEMP_DIR, f"morph_{timestamp}")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Batch or single processing mode
            if args.batch:
                # Batch processing mode
                video_extensions = ["mp4", "avi", "mov", "mkv", "webm"]
                
                # Let user select videos if not specified
                input_videos = []
                if args.input:
                    # Check if the input is a directory
                    if os.path.isdir(args.input):
                        # List all video files in the directory
                        for ext in video_extensions:
                            input_videos.extend(glob.glob(os.path.join(args.input, f"*.{ext}")))
                            input_videos.extend(glob.glob(os.path.join(args.input, f"*.{ext.upper()}")))
                    else:
                        # Assume it's a single file
                        input_videos = [args.input]
                else:
                    # List all video files in the SSD directory
                    video_files = list_files_with_extensions(SSD_DIR, video_extensions)
                    
                    if not video_files:
                        print("\nNo video files found in SSD folder.")
                        return
                    
                    print(f"\nFound {len(video_files)} video files in SSD folder.")
                    input_videos = select_files_from_list(
                        video_files,
                        "Select videos to process in batch:",
                        multiple=True
                    )
                
                if not input_videos:
                    print("No videos selected. Exiting.")
                    return
                
                # Process the selected videos
                output_dir = args.output_dir or SSD_DIR
                successful_videos = process_multiple_videos(input_videos, args, output_dir)
                
                # Update settings if processing was successful
                if successful_videos:
                    update_last_used_config(args)
                    
                    # Try to open the output folder
                    config = load_config()
                    if config["batch_processing"]["auto_open_result"] and successful_videos:
                        try:
                            if sys.platform == 'darwin':  # macOS
                                subprocess.run(['open', os.path.dirname(successful_videos[0])])
                            elif sys.platform == 'win32':  # Windows
                                subprocess.run(['explorer', os.path.dirname(successful_videos[0])])
                            elif sys.platform.startswith('linux'):  # Linux
                                subprocess.run(['xdg-open', os.path.dirname(successful_videos[0])])
                        except:
                            pass
            else:
                # Single video processing mode
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
                
                # Set up logging for this video
                logger = setup_logging(input_video)
                
                # Generate output filename if not provided
                output_video = args.output
                if not output_video:
                    output_video = generate_output_filename(f"_{os.path.splitext(os.path.basename(input_video))[0]}")
                    
                # Process the video
                result = process_single_video(input_video, output_video, args, logger)
                
                if result:
                    print("\n" + "="*60)
                    print("                 SUCCESS!")
                    print("="*60)
                    print(f"\nVideo with morphing effects created successfully:")
                    print(f"  {result}")
                    
                    # Update last used config
                    update_last_used_config(args)
                    
                    # Try to open the folder containing the video
                    try:
                        if sys.platform == 'darwin':  # macOS
                            subprocess.run(['open', os.path.dirname(result)])
                        elif sys.platform == 'win32':  # Windows
                            subprocess.run(['explorer', os.path.dirname(result)])
                        elif sys.platform.startswith('linux'):  # Linux
                            subprocess.run(['xdg-open', os.path.dirname(result)])
                    except:
                        pass
                else:
                    print("\nFailed to create output video.")
        finally:
            # Clean up temporary files if configured to do so
            config = load_config()
            if config["batch_processing"]["delete_temp_files"]:
                clean_up_temp_files(temp_dir)
                
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user. Exiting.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    batch_main()

# Add a helpful message at the end of the file
"""
Batch Video Morpher
------------------

How to use:

1. Basic usage: 
   python3 /Volumes/ssd1/batch_video_morpher.py

2. Batch processing:
   python3 /Volumes/ssd1/batch_video_morpher.py --batch

3. Using presets:
   python3 /Volumes/ssd1/batch_video_morpher.py --preset subtle

4. Saving presets:
   python3 /Volumes/ssd1/batch_video_morpher.py -k 7 -s 0.8 -t 20 --save-preset my_preset

5. List presets:
   python3 /Volumes/ssd1/batch_video_morpher.py --list-presets

6. Process multiple videos with specific settings:
   python3 /Volumes/ssd1/batch_video_morpher.py --batch -k 5 -s 0.7 -t 15 --scene-detection

7. Process videos in parallel:
   python3 /Volumes/ssd1/batch_video_morpher.py --batch --workers 4

8. Process a specific video with settings from last successful run:
   python3 /Volumes/ssd1/batch_video_morpher.py -i /path/to/video.mp4 --last-used
"""

#!/usr/bin/env python3
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Scale, Checkbutton, BooleanVar, Label, Button, Frame, StringVar, DoubleVar, Entry
from PIL import Image, ImageTk
import subprocess
from moviepy import VideoFileClip, AudioFileClip
import dlib
import time
from scipy.spatial import Delaunay
from scipy.interpolate import RectBivariateSpline
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

class MultiMorphApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Technique Morphing Video Creator")
        self.root.geometry("1000x800")
        
        # Variables
        self.image_paths = []
        self.audio_path = None
        self.output_path = "morphed_video.mp4"
        self.fps = 30
        self.duration_per_transition = 3.0
        
        # Morphing technique variables
        self.techniques = {
            "delaunay": {"name": "Delaunay Triangulation", "var": BooleanVar(value=True), "weight": DoubleVar(value=30)},
            "crossdissolve": {"name": "Cross-Dissolve", "var": BooleanVar(value=True), "weight": DoubleVar(value=20)},
            "opticalflow": {"name": "Optical Flow", "var": BooleanVar(value=True), "weight": DoubleVar(value=20)},
            "gridwarp": {"name": "Grid-based Warping", "var": BooleanVar(value=True), "weight": DoubleVar(value=15)},
            "frequency": {"name": "Frequency Domain", "var": BooleanVar(value=True), "weight": DoubleVar(value=15)}
        }
        
        # Face detector for landmark detection (used in Delaunay)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = None  # Will load on demand
        
        # Create UI
        self.create_ui()
    
    def create_ui(self):
        # Main frames
        top_frame = Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        image_frame = Frame(self.root)
        image_frame.pack(fill=tk.X, padx=10, pady=10)
        
        technique_frame = Frame(self.root)
        technique_frame.pack(fill=tk.X, padx=10, pady=10)
        
        settings_frame = Frame(self.root)
        settings_frame.pack(fill=tk.X, padx=10, pady=10)
        
        button_frame = Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=20)
        
        preview_frame = Frame(self.root)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top frame - Title and description
        Label(top_frame, text="Multi-Technique Morphing Video Creator", font=("Arial", 16, "bold")).pack(pady=5)
        Label(top_frame, text="Create morphing videos using multiple techniques simultaneously").pack()
        
        # Image selection
        Label(image_frame, text="Image Selection", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        image_buttons_frame = Frame(image_frame)
        image_buttons_frame.pack(fill=tk.X, pady=5)
        
        Button(image_buttons_frame, text="Select Images", command=self.select_images).pack(side=tk.LEFT, padx=5)
        Button(image_buttons_frame, text="Clear Images", command=self.clear_images).pack(side=tk.LEFT, padx=5)
        self.image_label = Label(image_frame, text="No images selected")
        self.image_label.pack(anchor=tk.W)
        
        # Audio selection
        audio_frame = Frame(image_frame)
        audio_frame.pack(fill=tk.X, pady=10)
        Button(audio_frame, text="Select Audio (Optional)", command=self.select_audio).pack(side=tk.LEFT, padx=5)
        self.audio_label = Label(audio_frame, text="No audio selected")
        self.audio_label.pack(side=tk.LEFT, padx=5)
        
        # Technique selection and weights
        Label(technique_frame, text="Morphing Techniques", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        Label(technique_frame, text="Select techniques and assign weights (total should equal 100%)").pack(anchor=tk.W)
        
        # Create technique checkboxes and weight sliders
        for technique_id, technique_info in self.techniques.items():
            technique_box = Frame(technique_frame, pady=5)
            technique_box.pack(fill=tk.X)
            
            Checkbutton(technique_box, text=technique_info["name"], 
                       variable=technique_info["var"],
                       command=self.update_weights).pack(side=tk.LEFT, padx=5)
            
            Label(technique_box, text="Weight:").pack(side=tk.LEFT, padx=5)
            
            weight_scale = Scale(technique_box, from_=0, to=100, 
                              orient=tk.HORIZONTAL, length=300,
                              variable=technique_info["weight"],
                              command=lambda _: self.update_weights())
            weight_scale.pack(side=tk.LEFT, padx=5)
            
            # Display percentage label
            technique_info["label"] = Label(technique_box, text="0%")
            technique_info["label"].pack(side=tk.LEFT, padx=5)
        
        # Settings
        Label(settings_frame, text="Settings", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        
        fps_frame = Frame(settings_frame)
        fps_frame.pack(fill=tk.X, pady=5)
        Label(fps_frame, text="Frames Per Second:").pack(side=tk.LEFT, padx=5)
        fps_entry = Entry(fps_frame, width=5)
        fps_entry.insert(0, str(self.fps))
        fps_entry.pack(side=tk.LEFT, padx=5)
        fps_entry.bind("<KeyRelease>", lambda e: self.update_fps(fps_entry.get()))
        
        duration_frame = Frame(settings_frame)
        duration_frame.pack(fill=tk.X, pady=5)
        Label(duration_frame, text="Seconds per Transition:").pack(side=tk.LEFT, padx=5)
        duration_entry = Entry(duration_frame, width=5)
        duration_entry.insert(0, str(self.duration_per_transition))
        duration_entry.pack(side=tk.LEFT, padx=5)
        duration_entry.bind("<KeyRelease>", lambda e: self.update_duration(duration_entry.get()))
        
        output_frame = Frame(settings_frame)
        output_frame.pack(fill=tk.X, pady=5)
        Label(output_frame, text="Output Filename:").pack(side=tk.LEFT, padx=5)
        output_entry = Entry(output_frame, width=30)
        output_entry.insert(0, self.output_path)
        output_entry.pack(side=tk.LEFT, padx=5)
        output_entry.bind("<KeyRelease>", lambda e: self.update_output_path(output_entry.get()))
        
        # Button frame
        self.total_weight_label = Label(button_frame, text="Total Weight: 0%", font=("Arial", 10, "bold"))
        self.total_weight_label.pack(pady=5)
        
        self.status_label = Label(button_frame, text="Ready", font=("Arial", 10))
        self.status_label.pack(pady=5)
        
        Button(button_frame, text="Normalize Weights", command=self.normalize_weights).pack(pady=5)
        Button(button_frame, text="Create Morphing Video", command=self.create_video).pack(pady=10)
        
        # Preview frame (for future implementation)
        Label(preview_frame, text="Preview will be shown here during processing", font=("Arial", 10, "italic")).pack()
        self.preview_canvas = tk.Canvas(preview_frame, bg="black", height=300)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Initialize weights
        self.update_weights()
    
    def select_images(self):
        filetypes = [("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        filenames = filedialog.askopenfilenames(title="Select Images", filetypes=filetypes)
        
        if filenames:
            self.image_paths = list(filenames)
            self.image_label.config(text=f"{len(self.image_paths)} images selected: {', '.join([os.path.basename(path) for path in self.image_paths])}")
    
    def clear_images(self):
        self.image_paths = []
        self.image_label.config(text="No images selected")
    
    def select_audio(self):
        filetypes = [("Audio files", "*.mp3 *.wav *.ogg *.m4a")]
        filename = filedialog.askopenfilename(title="Select Audio File", filetypes=filetypes)
        
        if filename:
            self.audio_path = filename
            self.audio_label.config(text=f"Selected: {os.path.basename(filename)}")
    
    def update_fps(self, value):
        try:
            self.fps = int(value)
        except ValueError:
            pass
    
    def update_duration(self, value):
        try:
            self.duration_per_transition = float(value)
        except ValueError:
            pass
    
    def update_output_path(self, value):
        if value:
            self.output_path = value
    
    def update_weights(self):
        total_weight = 0
        for technique_id, technique_info in self.techniques.items():
            if technique_info["var"].get():
                weight = technique_info["weight"].get()
                total_weight += weight
                technique_info["label"].config(text=f"{weight:.1f}%")
            else:
                technique_info["weight"].set(0)
                technique_info["label"].config(text="0%")
        
        self.total_weight_label.config(text=f"Total Weight: {total_weight:.1f}%")
        
        # Highlight in red if not 100%
        if abs(total_weight - 100) > 0.1:
            self.total_weight_label.config(fg="red")
        else:
            self.total_weight_label.config(fg="black")
    
    def normalize_weights(self):
        # Count enabled techniques
        enabled_techniques = [t for t_id, t in self.techniques.items() if t["var"].get()]
        
        if not enabled_techniques:
            self.status_label.config(text="Error: No techniques selected")
            return
            
        # Distribute weights evenly
        weight_per_technique = 100.0 / len(enabled_techniques)
        
        for technique_id, technique_info in self.techniques.items():
            if technique_info["var"].get():
                technique_info["weight"].set(weight_per_technique)
            else:
                technique_info["weight"].set(0)
                
        self.update_weights()
        self.status_label.config(text="Weights normalized")
    
    def create_video(self):
        if len(self.image_paths) < 2:
            self.status_label.config(text="Error: Need at least 2 images")
            return
        
        # Check if weights sum to 100%
        total_weight = sum(t["weight"].get() for t_id, t in self.techniques.items() if t["var"].get())
        if abs(total_weight - 100) > 0.1:
            self.status_label.config(text="Error: Weights must sum to 100%")
            return
        
        self.status_label.config(text="Loading images...")
        
        # Load images
        images = []
        for path in self.image_paths:
            img = cv2.imread(path)
            if img is None:
                self.status_label.config(text=f"Error: Could not load {os.path.basename(path)}")
                return
            images.append(img)
        
        # Make all images the same size
        target_height, target_width = 480, 640  # Standard size
        resized_images = []
        
        for img in images:
            resized = cv2.resize(img, (target_width, target_height))
            resized_images.append(resized)
        
        # Create morphing video
        self.status_label.config(text="Creating morphing video...")
        
        # Get selected techniques and their weights
        selected_techniques = {}
        for technique_id, technique_info in self.techniques.items():
            if technique_info["var"].get() and technique_info["weight"].get() > 0:
                selected_techniques[technique_id] = technique_info["weight"].get() / 100.0
        
        # If no technique is selected, use cross-dissolve by default
        if not selected_techniques:
            selected_techniques["crossdissolve"] = 1.0
        
        # Set up output video
        temp_output = "temp_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Calculate total frames
        total_frames_per_transition = int(self.fps * self.duration_per_transition)
        total_frames = total_frames_per_transition * (len(images) - 1)
        
        # Create video writer
        video_writer = cv2.VideoWriter(temp_output, fourcc, self.fps, (target_width, target_height))
        
        # Create frames for each transition
        frame_count = 0
        for i in range(len(images) - 1):
            self.status_label.config(text=f"Processing transition {i+1}/{len(images)-1}...")
            
            for frame in range(total_frames_per_transition):
                # Calculate alpha (0 to 1) for this frame
                alpha = frame / (total_frames_per_transition - 1)
                
                # Apply each selected morphing technique
                morphed_results = []
                
                for technique, weight in selected_techniques.items():
                    if technique == "delaunay":
                        morphed = self.delaunay_morph(resized_images[i], resized_images[i+1], alpha)
                    elif technique == "crossdissolve":
                        morphed = self.cross_dissolve(resized_images[i], resized_images[i+1], alpha)
                    elif technique == "opticalflow":
                        morphed = self.optical_flow_morph(resized_images[i], resized_images[i+1], alpha)
                    elif technique == "gridwarp":
                        morphed = self.grid_warp(resized_images[i], resized_images[i+1], alpha)
                    elif technique == "frequency":
                        morphed = self.frequency_domain_morph(resized_images[i], resized_images[i+1], alpha)
                    
                    morphed_results.append((morphed, weight))
                
                # Blend the results from different techniques
                blended_frame = self.blend_morphed_results(morphed_results)
                
                # Write frame to video
                video_writer.write(blended_frame)
                
                # Update preview
                if frame_count % 5 == 0:  # Update preview every 5 frames to reduce overhead
                    preview_img = cv2.cvtColor(blended_frame, cv2.COLOR_BGR2RGB)
                    preview_img = Image.fromarray(preview_img)
                    preview_img = ImageTk.PhotoImage(image=preview_img)
                    
                    # Keep a reference to prevent garbage collection
                    self.preview_img = preview_img
                    
                    # Display in canvas
                    self.preview_canvas.config(width=target_width, height=target_height)
                    self.preview_canvas.create_image(target_width//2, target_height//2, image=preview_img)
                    self.root.update()
                
                frame_count += 1
        
        # Release the video writer
        video_writer.release()
        
        # Add audio if selected
        if self.audio_path:
            self.status_label.config(text="Adding audio...")
            try:
                # Load video and audio clips
                video_clip = VideoFileClip(temp_output)
                audio_clip = AudioFileClip(self.audio_path)
                
                # If audio is longer than video, trim it
                if audio_clip.duration > video_clip.duration:
                    audio_clip = audio_clip.subclip(0, video_clip.duration)
                
                # Add audio to video
                final_clip = video_clip.set_audio(audio_clip)
                
                # Write the result
                final_clip.write_videofile(self.output_path, codec='libx264', audio_codec='aac')
                
                # Close the clips
                video_clip.close()
                audio_clip.close()
                
                # Remove the temporary file
                os.remove(temp_output)
            except Exception as e:
                self.status_label.config(text=f"Error adding audio: {str(e)}")
                # If audio fails, just rename the temp file
                os.rename(temp_output, self.output_path)
        else:
            # No audio, just rename the temp file
            os.rename(temp_output, self.output_path)
        
        self.status_label.config(text=f"Video created successfully: {self.output_path}")

    def delaunay_morph(self, img1, img2, alpha):
        """Morph between two images using Delaunay triangulation"""
        # Convert to grayscale for feature detection
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Get image dimensions
        h, w = gray1.shape[:2]
        
        # Detect facial landmarks or use general feature points
        try:
            # Try to use facial landmarks if faces are detected
            if self.predictor is None:
                # Load the predictor if not already loaded
                predictor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                             "shape_predictor_68_face_landmarks.dat")
                if os.path.exists(predictor_path):
                    self.predictor = dlib.shape_predictor(predictor_path)
                else:
                    raise FileNotFoundError("Facial landmark predictor file not found")
            
            # Detect faces
            faces1 = self.detector(gray1)
            faces2 = self.detector(gray2)
            
            if faces1 and faces2:
                # Get landmarks for first face in each image
                landmarks1 = self.predictor(gray1, faces1[0])
                landmarks2 = self.predictor(gray2, faces2[0])
                
                # Convert landmarks to points
                points1 = []
                points2 = []
                for i in range(68):  # 68 landmarks
                    points1.append((landmarks1.part(i).x, landmarks1.part(i).y))
                    points2.append((landmarks2.part(i).x, landmarks2.part(i).y))
            else:
                raise ValueError("No faces detected")
                
        except Exception as e:
            # Fall back to general feature detection
            feature_params = dict(maxCorners=80, qualityLevel=0.01, minDistance=10, blockSize=7)
            
            points1 = cv2.goodFeaturesToTrack(gray1, **feature_params)
            points2 = cv2.goodFeaturesToTrack(gray2, **feature_params)
            
            # Make sure we have the same number of points
            min_points = min(len(points1), len(points2))
            points1 = points1[:min_points]
            points2 = points2[:min_points]
            
            # Convert to list of tuples
            points1 = [(p[0][0], p[0][1]) for p in points1]
            points2 = [(p[0][0], p[0][1]) for p in points2]
        
        # Add corners and edge midpoints for better morphing
        for pt in [(0, 0), (0, h-1), (w-1, 0), (w-1, h-1),  # corners
                   (w//2, 0), (w//2, h-1), (0, h//2), (w-1, h//2)]:  # edge midpoints
            points1.append(pt)
            points2.append(pt)
        
        # Calculate intermediate points based on alpha
        points = []
        for i in range(len(points1)):
            x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
            y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
            points.append((int(x), int(y)))
        
        # Convert to numpy array for Delaunay
        points_array = np.array(points)
        
        # Create Delaunay triangulation
        tri = Delaunay(points_array)
        
        # Create output image
        output = np.zeros_like(img1)
        
        # For each triangle in the Delaunay triangulation
        for triangle in tri.simplices:
            # Get triangle vertices for all three images
            tri1 = [points1[triangle[0]], points1[triangle[1]], points1[triangle[2]]]
            tri2 = [points2[triangle[0]], points2[triangle[1]], points2[triangle[2]]]
            tri_morphed = [points[triangle[0]], points[triangle[1]], points[triangle[2]]]
            
            # Convert to numpy arrays
            tri1 = np.array(tri1, dtype=np.float32)
            tri2 = np.array(tri2, dtype=np.float32)
            tri_morphed = np.array(tri_morphed, dtype=np.float32)
            
            # Calculate bounding rectangle for the triangle
            rect = cv2.boundingRect(np.array([tri_morphed], dtype=np.float32))
            (x, y, rect_width, rect_height) = rect
            
            # Offset triangle coordinates
            tri1_rect = tri1 - np.array([x, y])
            tri2_rect = tri2 - np.array([x, y])
            tri_morphed_rect = tri_morphed - np.array([x, y])
            
            # Get the regions of interest in each image
            mask = np.zeros((rect_height, rect_width), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(tri_morphed_rect), 255)
            
            # Create affine transformations for both images
            warp_mat1 = cv2.getAffineTransform(tri1_rect, tri_morphed_rect)
            warp_mat2 = cv2.getAffineTransform(tri2_rect, tri_morphed_rect)
            
            # Warp the triangles
            img1_rect = cv2.warpAffine(img1[max(0, y):min(h, y+rect_height), 
                                           max(0, x):min(w, x+rect_width)],
                                    warp_mat1, (rect_width, rect_height))
            img2_rect = cv2.warpAffine(img2[max(0, y):min(h, y+rect_height), 
                                           max(0, x):min(w, x+rect_width)],
                                    warp_mat2, (rect_width, rect_height))
            
            # Alpha blend the warped triangles
            img_rect = (1 - alpha) * img1_rect + alpha * img2_rect
            
            # Copy the blended triangle to the output image
            output[max(0, y):min(h, y+rect_height), max(0, x):min(w, x+rect_width)][mask > 0] = \
                img_rect[mask > 0]
        
        return output.astype(np.uint8)

    def cross_dissolve(self, img1, img2, alpha):
        """Simple alpha blending between two images"""
        return cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)

    def optical_flow_morph(self, img1, img2, alpha):
        """Morph images using optical flow"""
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Get image dimensions
        h, w = img1.shape[:2]
        
        # Create meshgrid for coordinate mapping
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Scale the flow by alpha
        flow_x = alpha * flow[..., 0]
        flow_y = alpha * flow[..., 1]
        
        # Create maps for distortion
        map_x = (x + flow_x).astype(np.float32)
        map_y = (y + flow_y).astype(np.float32)
        
        # Warp first image
        warped_img1 = cv2.remap(img1, map_x, map_y, cv2.INTER_LINEAR)
        
        # Inverse flow for second image
        flow_x_inv = -alpha * flow[..., 0]
        flow_y_inv = -alpha * flow[..., 1]
        
        # Create inverse maps
        map_x_inv = (x + flow_x_inv).astype(np.float32)
        map_y_inv = (y + flow_y_inv).astype(np.float32)
        
        # Warp second image with inverse flow
        warped_img2 = cv2.remap(img2, map_x_inv, map_y_inv, cv2.INTER_LINEAR)
        
        # Cross-dissolve the warped images
        result = cv2.addWeighted(warped_img1, 1 - alpha, warped_img2, alpha, 0)
        
        return result

    def grid_warp(self, img1, img2, alpha):
        """Morph images using grid-based warping"""
        # Get image dimensions
        h, w = img1.shape[:2]
        
        # Create a regular grid
        grid_size = 20  # Size of grid cells
        x_points = np.arange(0, w, grid_size)
        y_points = np.arange(0, h, grid_size)
        
        # Ensure we include image boundaries
        if x_points[-1] < w - 1: 
            x_points = np.append(x_points, w - 1)
        if y_points[-1] < h - 1: 
            y_points = np.append(y_points, h - 1)
        
        # Create meshgrid
        x_grid, y_grid = np.meshgrid(x_points, y_points)
        
        # Flatten grid points
        grid_points = np.vstack((x_grid.flatten(), y_grid.flatten())).T
        
        # Add random offsets for second image grid (this creates the warping effect)
        np.random.seed(42)  # For reproducibility
        offsets = np.random.uniform(-grid_size/3, grid_size/3, size=grid_points.shape)
        
        # Calculate intermediate grid based on alpha
        morphed_grid = grid_points + alpha * offsets
        
        # Create a complete grid for the entire image
        x_full = np.arange(0, w)
        y_full = np.arange(0, h)
        x_grid_full, y_grid_full = np.meshgrid(x_full, y_full)
        
        # Create maps for both source images 
        map_x1 = x_grid_full.copy().astype(np.float32)
        map_y1 = y_grid_full.copy().astype(np.float32)
        
        map_x2 = x_grid_full.copy().astype(np.float32)
        map_y2 = y_grid_full.copy().astype(np.float32)
        
        # Interpolate the grid offsets for every pixel using thin plate spline or similar
        # Interpolate the grid offsets for every pixel using thin plate spline or similar
        # Here I'm using scipy's RectBivariateSpline for simplicity
        for channel in range(2):  # x and y channels
            # Reshape the grid point values for this channel
            grid_values = np.reshape(grid_points[:, channel], (len(y_points), len(x_points)))
            
            # Morph the grid point values
            morphed_values_1 = np.reshape(grid_points[:, channel] - alpha * offsets[:, channel], 
                                        (len(y_points), len(x_points)))
            morphed_values_2 = np.reshape(grid_points[:, channel] + (1-alpha) * offsets[:, channel], 
                                        (len(y_points), len(x_points)))
            
            # First image: from morphed grid to original grid
            spline = RectBivariateSpline(y_points, x_points, morphed_values_1)
            # Evaluate at each pixel coordinate in the full grid
            interp_vals = spline(y_full, x_full, grid=True)
            if channel == 0:
                map_x1 = interp_vals
            else:
                map_y1 = interp_vals
                
            # Second image: from morphed grid to warped grid
            spline = RectBivariateSpline(y_points, x_points, morphed_values_2)
            # Evaluate at each pixel coordinate in the full grid
            interp_vals = spline(y_full, x_full, grid=True)
            if channel == 0:
                map_x2 = interp_vals
            else:
                map_y2 = interp_vals
        # Apply the mapping
        warped_img1 = cv2.remap(img1, map_x1, map_y1, cv2.INTER_LINEAR)
        warped_img2 = cv2.remap(img2, map_x2, map_y2, cv2.INTER_LINEAR)
        
        # Cross-dissolve between the warped images
        result = cv2.addWeighted(warped_img1, 1-alpha, warped_img2, alpha, 0)
        
        return result

    def frequency_domain_morph(self, img1, img2, alpha):
        """Morph images in the frequency domain using Fourier transforms"""
        # Convert to grayscale for frequency domain processing
        # We'll only morph the intensity and keep color separate
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Apply Fourier Transform
        fft1 = fftshift(fft2(img1_gray))
        fft2 = fftshift(fft2(img2_gray))
        
        # Interpolate magnitude and phase
        magnitude1 = np.abs(fft1)
        magnitude2 = np.abs(fft2)
        phase1 = np.angle(fft1)
        phase2 = np.angle(fft2)
        
        # Linear interpolation of magnitude and phase
        morph_magnitude = (1 - alpha) * magnitude1 + alpha * magnitude2
        morph_phase = (1 - alpha) * phase1 + alpha * phase2
        
        # Reconstruct the morphed image
        morph_fft = morph_magnitude * np.exp(1j * morph_phase)
        morph_intensity = np.real(ifft2(ifftshift(morph_fft)))
        
        # Normalize the intensity to valid range [0, 255]
        morph_intensity = np.clip(morph_intensity, 0, 255)
        morph_intensity = ((morph_intensity - morph_intensity.min()) / 
                          (morph_intensity.max() - morph_intensity.min() + 1e-6) * 255).astype(np.uint8)
        
        # Now handle colors - simple cross-dissolve for color
        # Convert to HSV
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        
        # Interpolate Hue and Saturation
        h = (1 - alpha) * hsv1[:,:,0] + alpha * hsv2[:,:,0]
        s = (1 - alpha) * hsv1[:,:,1] + alpha * hsv2[:,:,1]
        
        # Create the morphed HSV image using the frequency-morphed intensity for V channel
        morphed_hsv = np.stack([h, s, morph_intensity], axis=2).astype(np.uint8)
        
        # Convert back to BGR
        result = cv2.cvtColor(morphed_hsv, cv2.COLOR_HSV2BGR)
        
        return result
    
    def blend_morphed_results(self, morphed_results):
        """
        Combine multiple morphed images based on their weights
        
        Parameters:
        - morphed_results: List of tuples (morphed_image, weight)
        
        Returns:
        - Blended image
        """
        if not morphed_results:
            raise ValueError("No morphed results provided")
        
        # Initialize with zeros
        result = np.zeros_like(morphed_results[0][0], dtype=np.float32)
        
        # Add each morphed image multiplied by its weight
        for morphed_img, weight in morphed_results:
            result += morphed_img.astype(np.float32) * weight
        
        # Normalize the result by total weight if multiple techniques are used
        total_weight = sum(weight for _, weight in morphed_results)
        if total_weight > 1.0:
            result = result / total_weight
            
        # Ensure the result is in valid range
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result

def main():
    """Run the Multi-Technique Morphing Video Creator application"""
    root = tk.Tk()
    app = MultiMorphApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

using Images, FileIO, WAV, VideoIO, Interpolations, ProgressMeter, Dates

# Function to check if a file exists
function file_exists(path)
    isfile(path)
end

# Function to check file extension
function has_extension(path, extension)
    endswith(lowercase(path), lowercase(extension))
end

# Function to list image files in Downloads folder
function list_image_files_in_downloads()
    download_folder = joinpath(homedir(), "Downloads")
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"]
    
    files = readdir(download_folder)
    image_files = filter(file -> any(ext -> endswith(lowercase(file), ext), image_extensions), files)
    
    if isempty(image_files)
        println("No image files found in Downloads folder.")
    else
        println("\nImage files found in Downloads folder:")
        for (i, file) in enumerate(image_files)
            println("$i. $file")
        end
        println("Enter numbers of images to use (comma-separated, e.g., 1,3,5):")
        println("NOTE: At least 2 images are required for morphing to work properly.")
    end
    
    return image_files
end

# Function to list audio files in Downloads folder
function list_audio_files_in_downloads()
    download_folder = joinpath(homedir(), "Downloads")
    audio_extensions = [".wav"]
    
    files = readdir(download_folder)
    audio_files = filter(file -> any(ext -> endswith(lowercase(file), ext), audio_extensions), files)
    
    if isempty(audio_files)
        println("No audio files found in Downloads folder.")
    else
        println("\nAudio files found in Downloads folder:")
        for (i, file) in enumerate(audio_files)
            println("$i. $file")
        end
        println("Enter number of audio file to use:")
    end
    
    return audio_files
end

# Function to generate a timestamped output filename
function generate_output_filename()
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    download_folder = joinpath(homedir(), "Downloads")
    return joinpath(download_folder, "morphed_video_$(timestamp).mp4")
end

# Function to create directory if it doesn't exist
function ensure_directory_exists(path)
    dir = dirname(path)
    if !isdir(dir)
        try
            mkpath(dir)
            println("Created directory: $dir")
            return true
        catch e
            println("Error creating directory $dir: $e")
            return false
        end
    end
    return true
end

# Function to prepend download folder path if needed
function with_download_path(path)
    download_folder = joinpath(homedir(), "Downloads")
    if startswith(path, "/") || startswith(path, "~")
        return path  # Already an absolute path
    else
        return joinpath(download_folder, path)  # Prepend download folder
    end
end

function create_morphing_video(image_paths::Vector{String}, audio_path::String, output_path::String, fps::Int)
    # Validate image paths
    if isempty(image_paths) || all(isempty, image_paths)
        error("No image paths provided. Please specify at least two image files.")
    end
    
    # Check if there are at least 2 images for morphing
    if length(image_paths) < 2
        error("At least 2 images are required for morphing to work properly. You selected $(length(image_paths)) image(s).")
    end
    
    # Check if image files exist
    for path in image_paths
        if !file_exists(path)
            error("Image file not found: $path")
        end
    end
    
    # Check if audio file exists and has correct extension
    if !file_exists(audio_path)
        error("Audio file not found: $audio_path")
    end
    
    if !has_extension(audio_path, ".wav")
        error("Audio file must be a .wav file: $audio_path")
    end
    
    # Ensure output directory exists
    if !ensure_directory_exists(output_path)
        error("Could not create output directory for: $output_path")
    end
    
    # Load images
    println("Loading images...")
    images = try
        [load(path) for path in image_paths]
    catch e
        error("Error loading images: $e")
    end
    num_images = length(images)

    # Load audio
    println("Loading audio file...")
    audio, sample_rate = try
        wavread(audio_path)
    catch e
        error("Error loading audio file: $e")
    end
    audio_duration = length(audio) / sample_rate

    # Calculate frame times
    frame_count = round(Int, audio_duration * fps)
    frame_times = range(0, audio_duration, length = frame_count)

    # Calculate image transition times
    image_duration = audio_duration / (num_images - 1)
    image_times = range(0, audio_duration, length = num_images)

    # Create video writer
    println("Setting up video writer...")
    writer = try
        size_first_image = size(images[1])
        # Use a more direct approach with FFMPEG encoder
        encoder_options = Dict(
            :color_range => 2,     # Full color range
            :crf => 23,            # Constant Rate Factor (quality - lower is better)
            :preset => "medium"    # Encoding speed/compression tradeoff
        )
        
        # Create the video writer with explicit properties
        VideoIO.openvideo(
            output_path,
            images[1],             # Use first image to determine format
            framerate=fps,         
            codec_name="libx264",  # Standard H.264 codec
            options=encoder_options,
            pixelformat=VideoIO.AV_PIX_FMT_YUV420P # Standard pixel format for compatibility
        )
    catch e
        error("Error creating video writer: $e")
    end

    @showprogress 1 "Creating video..." for t in frame_times
        # Determine current image indices
        image_index1 = findlast(x -> x <= t, image_times)
        image_index2 = image_index1 == num_images ? image_index1 : image_index1 + 1

        # Calculate interpolation factor
        if image_index2 == image_index1
          alpha = 0.0;
        else
          alpha = (t - image_times[image_index1]) / (image_times[image_index2] - image_times[image_index1])
        end

        # Morph images
        if num_images > 1
            interpolated_image = interpolate((images[image_index1], images[image_index2]), BSpline(Linear()))(alpha)
        else
          interpolated_image = images[1]
        end

        # Write frame to video
        VideoIO.write_frame(writer, interpolated_image)
    end

    # Close video writer and add audio
    VideoIO.close(writer)

    #add audio.
    println("Adding audio to video...")
    try
        run(`ffmpeg -y -i $output_path -i $audio_path -c:v copy -map 0:v:0 -map 1:a:0 $(replace(output_path, ".mp4", "_audio.mp4"))`)
        run(`rm $output_path`)
        run(`mv $(replace(output_path, ".mp4", "_audio.mp4")) $output_path`)
    catch e
        error("Error during FFmpeg processing: $e")
    end

    println("Video created successfully: $output_path")
end

# Get user input - Image selection
println("Listing available image files in Downloads folder...")
println("NOTE: You must select at least 2 images for the morphing process to work properly.")
available_images = list_image_files_in_downloads()

if isempty(available_images)
    error("No image files found in Downloads folder. Please add some images and try again.")
end

# Let user select images by number
image_numbers_str = readline()
image_numbers = try
    parse.(Int, filter(!isempty, strip.(split(image_numbers_str, ","))))
catch e
    error("Invalid input. Please enter numbers separated by commas.")
end

# Validate image selections
if isempty(image_numbers)
    error("No images selected. Please select at least two images.")
end

# Check if user selected at least 2 images
if length(image_numbers) < 2
    error("At least 2 images are required for morphing to work properly. You selected $(length(image_numbers)) image(s).")
end

download_folder = joinpath(homedir(), "Downloads")
image_paths = String[]

for num in image_numbers
    if num < 1 || num > length(available_images)
        error("Invalid image number: $num. Please select numbers between 1 and $(length(available_images)).")
    end
    push!(image_paths, joinpath(download_folder, available_images[num]))
end

println("Selected $(length(image_paths)) images.")

# Audio file selection
println("\nListing available audio files in Downloads folder...")
available_audio = list_audio_files_in_downloads()

if isempty(available_audio)
    error("No .wav audio files found in Downloads folder. Please add some .wav files and try again.")
end

# Let user select audio by number
audio_number_str = readline()
audio_number = try
    parse(Int, strip(audio_number_str))
catch e
    error("Invalid input. Please enter a single number.")
end

# Validate audio selection
if audio_number < 1 || audio_number > length(available_audio)
    error("Invalid audio number: $audio_number. Please select a number between 1 and $(length(available_audio)).")
end

audio_path = joinpath(download_folder, available_audio[audio_number])
println("Selected audio: $(available_audio[audio_number])")

# Generate automatic output filename
output_path = generate_output_filename()
println("\nAutomatic output path: $output_path")
# Output path is already generated with .mp4 extension

# Ensure output directory exists
ensure_directory_exists(output_path)

println("Enter frames per second (fps) [press Enter for default of 30]:")
fps_input = readline()

# Validate fps
fps = try
    if isempty(strip(fps_input))
        println("Using default value of 30 FPS.")
        30
    else
        fps_value = parse(Int, fps_input)
        if fps_value <= 0
            println("Warning: FPS must be positive. Using default value of 30.")
            30
        else
            fps_value
        end
    end
catch e
    println("Invalid FPS value. Using default value of 30.")
    30
end

println("\nSummary:")
println("- Images ($(length(image_paths))): $(join(basename.(image_paths), ", "))")
println("- Audio: $(basename(audio_path))")
println("- Output: $(basename(output_path))")
println("- FPS: $fps")
println("\nStarting video creation... This may take some time depending on the number of images and length of audio.")

# Create morphing video
try
    create_morphing_video(image_paths, audio_path, output_path, fps)
catch e
    println("\nError creating video: $e")
    exit(1)
end

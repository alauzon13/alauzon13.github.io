import os
from rembg import remove
from PIL import Image
import uuid

def remove_background_from_images(input_folder, output_folder):
    """
    Removes the background from images in the input folder and saves them to the output folder.

    Args:
        input_folder (str): Path to the input directory containing images.
        output_folder (str): Path to the output directory where processed images will be saved.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Supported image extensions
    valid_extensions = ['.png', '.jpg', '.jpeg', '.webp']

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            input_path = os.path.join(input_folder, filename)
            
            # Generate a unique identifier for the file
            unique_id = str(uuid.uuid4())
            output_path = os.path.join(output_folder, unique_id + '.png')

            try:
                # Open the image
                input_image = Image.open(input_path)

                # Remove background
                output_image = remove(input_image)

                # Save the output image as PNG (with transparency)
                output_image.save(output_path)
                print(f"Processed: {filename} -> {unique_id}.png")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

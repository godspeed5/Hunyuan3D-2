from typing import List, Union
from PIL import Image
import torch
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

# Global variables for the model and device
model = None
device = None

def initialize_model(pretrained_model: str = "tencent/Hunyuan3D-2", device_str: str = "cuda"):
    """
    Initialize the global Hunyuan 3D model for processing.

    Parameters:
        pretrained_model (str): The pretrained model's identifier or path.
        device_str (str): The device to load the model on ("cuda" or "cpu").

    Raises:
        ValueError: If an invalid device is specified or if CUDA is unavailable when requested.
    """
    global model, device

    device = device_str
    if not torch.cuda.is_available() and device == "cuda":
        print("CUDA is not available. Falling back to CPU.")
        device = "cpu"

    if model is None:
        print(f"Loading model {pretrained_model} on {device}...")
        model = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(pretrained_model)
        model.to(device)
        print("Model loaded successfully.")



def process_images(
    images: List[Union[Image.Image, str]],
    output_path: str = "output.glb",
    return_glb: bool = False
):
    """
    Converts a list of images into a 3D model and saves it as a GLB file.

    Parameters:
        images (List[Union[Image.Image, str]]): A list of input images or file paths.
        output_path (str): The file path to save the resulting GLB file.
        return_glb (bool): If True, returns the GLB binary data instead of saving to file.

    Returns:
        bytes: The GLB binary data if `return_glb` is True; otherwise, None.
    """
    if model is None:
        raise RuntimeError("Model is not initialized. Please call `initialize_model` first.")

    # Load and preprocess images
    preprocessed_images = []
    for img in images:
        if isinstance(img, str):
            img = Image.open(img)
        preprocessed_images.append(img)

    # Process each image through the pipeline
    glb_data = None
    for image in preprocessed_images:
        print(f"Processing image {image} into a 3D model...")
        mesh = model(image=image)[0]

        # Save or collect GLB
        glb_data = mesh.export_to_glb()
        if not return_glb:
            with open(output_path, "wb") as f:
                f.write(glb_data)
            print(f"3D model saved to {output_path}.")
    
    if return_glb:
        return glb_data

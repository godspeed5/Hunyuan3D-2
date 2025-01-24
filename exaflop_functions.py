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
    images: Union[str, List[str], Image.Image],
    output_path: str = "output.glb",
    return_glb: bool = False
):
    """
    Converts a list of images into a 3D model and saves it as a GLB file.

    Parameters:
        images (List[Union[Image.Image, str]]): A list of input images or file paths. Currently supports only 1 image. Extend to multiple views later
        output_path (str): The file path to save the resulting GLB file.
        return_glb (bool): If True, returns the GLB binary data instead of saving to file.

    Returns:
        bytes: The GLB binary data if `return_glb` is True; otherwise, None.
    """
    if model is None:
        raise RuntimeError("Model is not initialized. Please call `initialize_model` first.")

    print(f"Processing images into a 3D model...")
    mesh = model(image=images)[0]

    # Save or collect GLB
    mesh.export(output_path)

if __name__ == "__main__":
    # Initialize the model
    initialize_model()

    # Process images
    process_images("/workspace/Hunyuan3D-2/hy3dgen/front.png", output_path="output_front.glb")
    process_images(["/workspace/Hunyuan3D-2/hy3dgen/side.png", "/workspace/Hunyuan3D-2/hy3dgen/front.png", "/workspace/Hunyuan3D-2/hy3dgen/back.png", "/workspace/Hunyuan3D-2/side_2.png"], output_path="output_all_views.glb")

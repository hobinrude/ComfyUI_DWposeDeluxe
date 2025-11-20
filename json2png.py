# python workflow2png.py --image_path "/path/to/image.png" --workflow_path "/path/to/workflow.json"

import argparse
import os
from PIL import Image, PngImagePlugin

def make_workflow_png(image_path, workflow_path):
    if not os.path.exists(image_path):
        raise ValueError(f"Invalid image path `{image_path}`")
    if not os.path.exists(workflow_path):
        raise ValueError(f"Invalid workflow path `{workflow_path}`")
    path = os.path.dirname(image_path)
    filename = os.path.basename(image_path).rsplit('.', 1)[0]

    data = None # Initialize data here
    try:
        with open(workflow_path, "r") as file:
            data = file.read()
    except OSError as e:
        # It's better to raise the error here so the program doesn't proceed with missing data
        raise Exception(f"Error reading the workflow JSON from {workflow_path}: {e}")

    image = Image.open(image_path)
    info = PngImagePlugin.PngInfo()

    if data is not None: # Only add metadata if data was successfully read
        info.add_text("workflow", data)
    else:
        # This branch should ideally not be reached if the exception is raised above,
        # but it's good for robustness if the exception handling changes.
        print(f"Warning: Workflow data could not be read from {workflow_path}. Skipping metadata addition.")

    new_path = os.path.join(path, filename+'.png')
    image.save(new_path, "PNG", pnginfo=info)
    return new_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add workflow metadata to a PNG image")
    parser.add_argument("--image_path", type=str, help="Path to the PNG image")
    parser.add_argument("--workflow_path", type=str, help="Path to the workflow JSON file")
    args = parser.parse_args()
    new_image_path = make_workflow_png(args.image_path, args.workflow_path)
    
    print(f"Workflow added to `{new_image_path}`")
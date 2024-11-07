import requests
import os
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def setup_model():
    """Initialize the model and processor"""
    model_id = "IDEA-Research/grounding-dino-tiny"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    return processor, model, device

def load_image(image_path):
    """Load and verify image"""
    try:
        image = Image.open(image_path)
        print(f"Successfully loaded: {os.path.basename(image_path)}")
        print(f"Image size: {image.size}")
        return image
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return None

def detect_objects(image, processor, model, device):
    """Perform object detection"""
    # Detailed text prompts for better detection
    text = """
    a warehouse floor or ground surface made of concrete. 
    a wooden or plastic storage pallet used for storing goods.
    a warehouse storage rack or shelf system.
    """

    # Process image and text
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process results
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.3,  # Adjusted for better detection
        text_threshold=0.25,
        target_sizes=[image.size[::-1]]
    )
    
    return results

def visualize_results(image, results):
    """Visualize detection results on the image"""
    # Create a copy of the image for drawing
    draw = ImageDraw.Draw(image)
    
    # Define colors for different classes
    colors = {
        "ground": "red",
        "pallet": "blue",
        "rack": "green"
    }
    
    # Draw boxes and labels
    for score, label, box in zip(
        results[0]["scores"], 
        results[0]["labels"], 
        results[0]["boxes"]
    ):
        # Convert box coordinates to integers
        box = [round(i) for i in box.tolist()]
        
        # Get color based on label
        color = colors.get(label, "yellow")
        
        # Draw bounding box
        draw.rectangle(box, outline=color, width=3)
        
        # Draw label and score
        label_text = f"{label}: {score:.2f}"
        draw.text((box[0], box[1] - 20), label_text, fill=color)
    
    return image

def save_results(image, results, output_path="./output"):
    """Save annotated image and detection results"""
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save annotated image
    annotated_image = visualize_results(image.copy(), results)
    output_file = os.path.join(output_path, "annotated_pallet.jpg")
    annotated_image.save(output_file)
    print(f"Saved annotated image to: {output_file}")
    
    # Save detection results to text file
    results_file = os.path.join(output_path, "detection_results.txt")
    with open(results_file, "w") as f:
        f.write("Detection Results:\n")
        for score, label, box in zip(
            results[0]["scores"], 
            results[0]["labels"], 
            results[0]["boxes"]
        ):
            f.write(f"Label: {label}, Score: {score:.2f}, Box: {box.tolist()}\n")
    print(f"Saved detection results to: {results_file}")

def main():
    # Setup
    processor, model, device = setup_model()
    
    # Load image
    image_path = "./data/Pallets/pallet_12.jpg"  # Adjust path as needed
    image = load_image(image_path)
    
    if image is None:
        return
    
    # Perform detection
    results = detect_objects(image, processor, model, device)
    
    # Save results
    save_results(image, results)
    
    # Print detection summary
    print("\nDetection Summary:")
    for score, label in zip(results[0]["scores"], results[0]["labels"]):
        print(f"Found {label} with confidence: {score:.2f}")

if __name__ == "__main__":
    main()
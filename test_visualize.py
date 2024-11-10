import torch
import torchvision.transforms as T
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from PIL import Image

def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    return model

def prepare_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)
    return img_tensor, image

def run_inference(model, img_tensor, device):
    if torch.cuda.is_available():
        model = model.to(device)
        img_tensor = img_tensor.to(device)
    with torch.no_grad():
        predictions = model(img_tensor)
    return predictions

def draw_bounding_boxes(image, boxes, labels, scores, class_names, threshold=0.5):
    image = np.array(image)
    for i in range(boxes.shape[0]):  # Use shape[0] to get number of boxes
        if scores[i] > threshold:
            box = boxes[i].cpu().numpy().astype(int)  # Ensure boxes are numpy arrays
            label = labels[i].item()  # Convert to native Python type
            score = scores[i].item()  # Convert to native Python type
            class_name = class_names[label]  # Map label to class name
            
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(image, f"Class: {class_name}, Score: {score:.2f}", 
                        (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return image

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    model = load_model(args.model, device)
    
    # Define the image transformation
    transform = T.Compose([T.Resize((800, 800)), T.ToTensor()])
    
    # Prepare the image
    img_tensor, img = prepare_image(args.image, transform)
    
    # Run inference
    predictions = run_inference(model, img_tensor, device)

    # Extract predictions (adjust based on the log format)
    scores = predictions[0]['scores']  # Class scores
    labels = predictions[0]['labels']  # Labels
    boxes = predictions[0]['boxes']    # Bounding boxes
    
    # Convert boxes, labels, and scores to numpy arrays for easier processing
    boxes = boxes.cpu().numpy()  # Convert boxes to NumPy array
    labels = labels.cpu().numpy()  # Convert labels to NumPy array
    scores = scores.cpu().numpy()  # Convert scores to NumPy array
    
    # Class mapping (including custom classes)
    class_names = {0: 'background', 1: 'fire hydrant', 2: 'emergency pole'}  # Map label to class names
    
    # Draw bounding boxes on the image
    output_image = draw_bounding_boxes(img, boxes, labels, scores, class_names)
    
    # Convert from RGB to BGR for OpenCV
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    
    # Display the image
    plt.imshow(output_image)
    plt.axis('off')  # Remove axis
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize object detection results")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--num_classes", type=int, required=True, help="Total number of classes including custom classes")
    args = parser.parse_args()
    
    main(args)

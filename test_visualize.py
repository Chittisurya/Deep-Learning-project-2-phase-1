import torch
import torchvision.transforms as T
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from PIL import Image

def load_model(model_path):
    model = torch.load(model_path)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    return model

def prepare_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)
    return img_tensor, image

def run_inference(model, img_tensor):
    if torch.cuda.is_available():
        model = model.cuda()
        img_tensor = img_tensor.cuda()
    with torch.no_grad():
        predictions = model(img_tensor)
    return predictions

def draw_bounding_boxes(image, boxes, labels, scores, threshold=0.5):
    image = np.array(image)
    for i in range(len(boxes)):
        if scores[i] > threshold:
            box = boxes[i].cpu().numpy().astype(int)
            label = labels[i].item()
            score = scores[i].item()
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(image, f"Class: {label}, Score: {score:.2f}", 
                        (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return image

def main(args):
    model = load_model(args.model)
    transform = T.Compose([T.Resize((800, 800)), T.ToTensor()])
    img_tensor, img = prepare_image(args.image, transform)
    predictions = run_inference(model, img_tensor)

    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']

    output_image = draw_bounding_boxes(img, boxes, labels, scores)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    plt.imshow(output_image)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize object detection results")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--num_classes", type=int, required=True, help="Total number of classes including custom classes")
    args = parser.parse_args()
    main(args)

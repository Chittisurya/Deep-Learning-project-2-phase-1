import torch
import numpy as np
import time
import cv2
import csv
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
sys.path.append(os.path.abspath("/content/Deep-Learning-project-2-phase-1"))

def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


# Draws a caption above the box in an image
def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def initialize_model(model_path, class_list):
    import torch
    from retinanet import model

    # Load class labels
    with open(class_list, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    num_classes = len(labels)

    # Define the model architecture
    retinanet = model.resnet50(num_classes=num_classes, pretrained=False)

    # Load the state dictionary
    state_dict = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

    # Load the weights into the model
    retinanet.load_state_dict(state_dict)

    # Move the model to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    retinanet = retinanet.to(device)
    retinanet.eval()  # Set the model to evaluation mode

    return retinanet, labels, device



# Prediction and visualization function
def detect_and_visualize(image, model, labels, device, threshold=0.5):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_orig = image_cv.copy()
    rows, cols, cns = image_cv.shape

    # Rescale the image (keep aspect ratio)
    min_side = 608
    max_side = 1024
    smallest_side = min(rows, cols)
    scale = min_side / smallest_side
    largest_side = max(rows, cols)

    if largest_side * scale > max_side:
        scale = max_side / largest_side

    image_cv = cv2.resize(image_cv, (int(round(cols * scale)), int(round(rows * scale))))
    rows, cols, cns = image_cv.shape

    # Padding to make dimensions divisible by 32
    pad_w = 32 - rows % 32
    pad_h = 32 - cols % 32

    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = image_cv.astype(np.float32)
    image_cv = new_image.astype(np.float32)
    image_cv /= 255
    image_cv -= [0.485, 0.456, 0.406]
    image_cv /= [0.229, 0.224, 0.225]
    image_cv = np.expand_dims(image_cv, 0)
    image_cv = np.transpose(image_cv, (0, 3, 1, 2))

    with torch.no_grad():
        image_tensor = torch.from_numpy(image_cv).to(device)

        st = time.time()
        scores, classification, transformed_anchors = model(image_tensor.float())
        print(f'Elapsed time: {time.time() - st}')
        
        idxs = np.where(scores.cpu() > threshold)

        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :]

            x1 = int(bbox[0] / scale)
            y1 = int(bbox[1] / scale)
            x2 = int(bbox[2] / scale)
            y2 = int(bbox[3] / scale)
            label_name = labels[int(classification[idxs[0][j]])]
            score = scores[idxs[0][j]]
            caption = f'{label_name} {score:.3f}'
            draw_caption(image_orig, (x1, y1, x2, y2), caption)
            cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

    # Convert BGR to RGB for Gradio
    result_image = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_image)


# Gradio interface function
def gradio_interface(image, model_path, class_list, threshold=0.5):
    global model, labels, device
    if model is None or labels is None or device is None:
        model, labels, device = initialize_model(model_path, class_list)

    return detect_and_visualize(image, model, labels, device, threshold)


# Global variables for model, labels, and device
model, labels, device = None, None, None

# Gradio interface
interface = gr.Interface(
    fn=lambda image, threshold: gradio_interface(image, "model_final_gradio_v2.pth", "classes.csv", threshold),
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Slider(0.0, 1.0, value=0.5, label="Confidence Threshold"),
    ],
    outputs=gr.Image(type="pil", label="Output Image"),
    title="Object Detection with RetinaNet",
    description="Upload an image, and the model will detect objects and annotate them with bounding boxes."
).launch(share=True)

if __name__ == "__main__":
    interface.launch()

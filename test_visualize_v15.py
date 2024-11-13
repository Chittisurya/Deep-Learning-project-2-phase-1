import torch
import numpy as np
import time
import os
import csv
import cv2
import argparse


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


def detect_image(image_path, model_path, class_list):

    # Load class names from CSV
    with open(class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))

    # Reverse the dictionary to get the class name by id
    labels = {v: k for k, v in classes.items()}

    # Load the model
    print(f"Loading model from {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Choose device (GPU or CPU)
    model = torch.load(model_path, map_location=device)  # Load model and map it to the correct device
    
    if torch.cuda.device_count() > 1:  # If using multiple GPUs
        model = torch.nn.DataParallel(model)

    model.to(device)  # Ensure the model is on the correct device
    model.eval()

    # Loop through images in the provided directory
    for img_name in os.listdir(image_path):
        image = cv2.imread(os.path.join(image_path, img_name))
        if image is None:
            continue

        image_orig = image.copy()
        rows, cols, cns = image.shape

        # Rescale the image (keep aspect ratio)
        min_side = 608
        max_side = 1024
        smallest_side = min(rows, cols)
        scale = min_side / smallest_side
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        image = cv2.resize(image, (int(round(cols * scale)), int(round(rows * scale))))
        rows, cols, cns = image.shape

        # Padding to make dimensions divisible by 32
        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        image = new_image.astype(np.float32)
        image /= 255
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))

        with torch.no_grad():
            image = torch.from_numpy(image).to(device)  # Move image to the same device as the model

            st = time.time()
            # Get predictions from the model
            scores, classification, transformed_anchors = model(image.float())
            print(f'Elapsed time: {time.time() - st}')
            print(classification)
            idxs = np.where(scores.cpu() > 0.5)
            print(idxs)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]

                x1 = int(bbox[0] / scale)
                y1 = int(bbox[1] / scale)
                x2 = int(bbox[2] / scale)
                y2 = int(bbox[3] / scale)
                label_name = labels[int(classification[idxs[0][j]])]
                print(bbox, classification.shape)
                score = scores[j]
                caption = f'{label_name} {score:.3f}'
                draw_caption(image_orig, (x1, y1, x2, y2), caption)
                cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

            # Save the processed image
            cv2.imwrite(img_name, image_orig)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')

    parser.add_argument('--image_dir', help='Path to directory containing images')
    parser.add_argument('--model_path', help='Path to model')
    parser.add_argument('--class_list', help='Path to CSV file listing class names (see README)')

    args = parser.parse_args()

    detect_image(args.image_dir, args.model_path, args.class_list)

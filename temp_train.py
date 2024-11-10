import argparse
import os
import json
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn
from torchvision.models.detection import retinanet_resnet50_fpn, create_default_roi_box_predictor

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.transform = transform
        self.annotations = self.load_annotations()

    def load_annotations(self):
        with open(self.annotation_file) as f:
            return json.load(f)  # Assuming COCO-style JSON format

    def __len__(self):
        return len(self.annotations['images'])

    def __getitem__(self, idx):
        img_info = self.annotations['images'][idx]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        boxes = self.annotations['annotations'][idx]['bbox']
        labels = self.annotations['annotations'][idx]['category_id']
        
        if self.transform:
            image = self.transform(image)
        
        return image, boxes, labels

# Data augmentation and normalization pipeline
def get_transforms():
    return T.Compose([
        T.ToTensor(),
        T.Resize((800, 800)),  # Resize as needed
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # COCO mean/std
        T.RandomHorizontalFlip(),
        T.RandomRotation(30),
    ])

# Function to load dataset and partition it
def partition_dataset(image_dir, annotation_file):
    with open(annotation_file) as f:
        annotations = json.load(f)
    
    images = annotations['images']
    train_images, temp_images = train_test_split(images, test_size=0.2, random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)

    return train_images, val_images, test_images

# Modify RetinaNet's output layer
def modify_model(model, num_classes):
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = create_default_roi_box_predictor(in_features, num_classes=num_classes)
    return model

# Training loop
def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    running_loss = 0
    for images, targets in train_loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    return running_loss / len(train_loader)

# Validation loop
def evaluate(model, val_loader, device):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            running_loss += losses.item()

    return running_loss / len(val_loader)

# Visualize predictions
def visualize_predictions(model, loader, device):
    model.eval()
    images, targets = next(iter(loader))
    images = [image.to(device) for image in images]

    with torch.no_grad():
        predictions = model(images)

    for i, image in enumerate(images):
        plt.imshow(image.cpu().permute(1, 2, 0))  # Convert tensor to numpy
        plt.title("Predictions")
        boxes = predictions[i]['boxes'].cpu().numpy()
        labels = predictions[i]['labels'].cpu().numpy()
        for box, label in zip(boxes, labels):
            plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none'))
        plt.show()

# Main function to run everything
def main(args):
    # Load dataset and partition
    train_images, val_images, test_images = partition_dataset(args.image_dir, args.annotation_file)

    # Prepare DataLoader
    transform = get_transforms()
    train_dataset = CustomDataset(args.image_dir, args.annotation_file, transform=transform)
    val_dataset = CustomDataset(args.image_dir, args.annotation_file, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Load the pre-trained RetinaNet model
    model = retinanet_resnet50_fpn(pretrained=True)

    # Modify the model's output layer to account for the new classes
    model = modify_model(model, num_classes=82)  # 80 COCO classes + 2 new classes

    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Move model to the appropriate device (GPU or CPU)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Train and evaluate
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss}")
    
    # Visualize predictions
    visualize_predictions(model, train_loader, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune RetinaNet on a custom dataset with two new classes.")
    
    # Arguments for dataset and training
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the directory containing images.')
    parser.add_argument('--annotation_file', type=str, required=True, help='Path to the COCO-style annotation JSON file.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and evaluation.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model.')

    args = parser.parse_args()

    main(args)

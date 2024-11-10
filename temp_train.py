import argparse
import collections
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from torch.utils.data import DataLoader
from retinanet import coco_eval
from retinanet import csv_eval

print('CUDA available: {}'.format(torch.cuda.is_available()))

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    
    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--num_classes', type=int, default=80, help='Total number of classes, including custom classes.')

    parser = parser.parse_args(args)

    # Create the data loaders
    if parser.dataset == 'coco':
        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO.')
        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':
        if parser.csv_train is None or parser.csv_classes is None:
            raise ValueError('Must provide --csv_train and --csv_classes for CSV dataset.')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = None if parser.csv_val is None else CSVDataset(
            train_file=parser.csv_val, class_list=parser.csv_classes,
            transform=transforms.Compose([Normalizer(), Resizer()]))
    else:
        raise ValueError('Dataset type not understood (must be csv or coco).')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    model_func = getattr(model, f'resnet{parser.depth}', None)
    if not model_func:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    
    retinanet = model_func(num_classes=parser.num_classes, pretrained=True)

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    
    retinanet.train()
    retinanet.module.freeze_bn()

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    loss_hist = collections.deque(maxlen=500)

    print('Num training images:', len(dataset_train))

    for epoch_num in range(parser.epochs):
        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []
        for iter_num, data in enumerate(dataloader_train):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
            else:
                classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                
            loss = classification_loss.mean() + regression_loss.mean()
            if loss == 0:
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
            optimizer.step()
            loss_hist.append(float(loss))
            epoch_loss.append(float(loss))

            print(f"Epoch: {epoch_num} | Iter: {iter_num} | Cls loss: {classification_loss.mean():.5f} | "
                  f"Reg loss: {regression_loss.mean():.5f} | Avg loss: {np.mean(loss_hist):.5f}")

            del classification_loss, regression_loss

        # Evaluation and prediction formatting
        if parser.dataset == 'coco':
            coco_eval.evaluate_coco(dataset_val, retinanet)
        elif parser.dataset == 'csv' and parser.csv_val:
            csv_eval.evaluate(dataset_val, retinanet)

        scheduler.step(np.mean(epoch_loss))

        # Save model
        torch.save(retinanet.module, f'{parser.dataset}_retinanet_{epoch_num}.pt')

    torch.save(retinanet, 'model_final.pt')


# Add a helper function to format predictions
def format_predictions(predictions):
    formatted_predictions = {
        'boxes': predictions[0]['boxes'].cpu().numpy(),
        'labels': predictions[0]['labels'].cpu().numpy(),
        'scores': predictions[0]['scores'].cpu().numpy()
    }
    return formatted_predictions


if __name__ == '__main__':
    main()

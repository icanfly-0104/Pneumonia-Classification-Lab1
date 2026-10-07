'''
Train the model

- Plot the training result (including validation info).
- Output the model weights.
'''
import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import seaborn as sns
import CustomDataLoader
import copy

def measurement(outputs, labels, smooth=1e-10):
    tp, tn, fp, fn = smooth, smooth, smooth, smooth
    labels = labels.cpu().numpy()
    outputs = outputs.detach().cpu().clone().numpy()
    for j in range(labels.shape[0]):
        if (int(outputs[j]) == 1 and int(labels[j]) == 1):
            tp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 0):
            tn += 1
        if (int(outputs[j]) == 1 and int(labels[j]) == 0):
            fp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 1):
            fn += 1
    return tp, tn, fp, fn

def plot_accuracy(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig('accuracy_curve.png')
    plt.show()

def plot_f1_score(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history['val_recall'], label='Recall')
    plt.plot(history['val_precision'], label='Precision')
    plt.plot(history['val_f1'], label='F1')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()
    plt.grid()
    plt.savefig('validation_metrics.png')
    plt.show()

def plot_loss(history):
    # Loss Graph
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig('loss_curve.png')
    plt.show()

def plot_confusion_matrix(c_matrix):
    # Converto NumPy array
    cm = np.array(c_matrix)

    # Plot
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['NORMAL', 'PNEUMONIA'],
        yticklabels=['NORMAL', 'PNEUMONIA'],
        annot_kws={'size': 16}
    )

    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('Actual Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=16)

    plt.tight_layout()

    # Save image
    plt.savefig('confusion_matrix.png',dpi=300,bbox_inches='tight')

    plt.show()
    plt.close()

def train(device, train_loader, val_loader, model, criterion, optimizer):

    best_acc = 0.0
    best_f1_score = 0.0
    best_loss = 100
    best_model_wts = None

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_recall': [],
        'val_precision': [],
        'val_f1': []
    }

    for epoch in range(1, args.num_epochs + 1):

        # =========================
        # Training
        # =========================

        model.train()

        train_loss = 0.0
        tp, tn, fp, fn = 0, 0, 0, 0

        for inputs, labels in tqdm(train_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # Loss
            train_loss += loss.item() * inputs.size(0)

            # Prediction
            predictions = torch.max(outputs, 1).indices

            sub_tp, sub_tn, sub_fp, sub_fn = \
                measurement(predictions, labels)

            tp += sub_tp
            tn += sub_tn
            fp += sub_fp
            fn += sub_fn

        train_loss /= len(train_loader.dataset)

        train_acc = (
            (tp + tn) /
            (tp + tn + fp + fn)
            * 100
        )

        print(f'Epoch: {epoch}')
        print(f'↳ Train Loss: {train_loss:.4f}')
        print(f'↳ Training Acc.(%): {train_acc:.2f}%')

        # =========================
        # Validation
        # =========================

        val_loss, val_acc, val_recall, \
        val_precision, val_f1 = validation(val_loader,model,criterion)

        # =========================
        # Save history
        # =========================

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        history['val_recall'].append(val_recall)
        history['val_precision'].append(val_precision)
        history['val_f1'].append(val_f1)

        # =========================
        # Save best model
        # =========================

        if val_acc >= best_acc:
            best_acc = val_acc
            if val_loss <= best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        
            if val_f1 >= best_f1_score:
                best_f1_score = val_f1

    # =========================
    # Load best model
    # =========================

    model.load_state_dict(best_model_wts)

    torch.save(model.state_dict(), 'model_weights.pt')

    return best_acc, best_f1_score, history

def validation(val_loader, model, criterion):
    val_acc = 0.0
    val_loss = 0.0

    tp, tn, fp, fn = 0, 0, 0, 0

    model.eval()

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Model output
            outputs = model(images)

            # Validation loss
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            # Prediction
            predictions = torch.max(outputs, 1).indices

            # Confusion matrix
            sub_tp, sub_tn, sub_fp, sub_fn = \
                measurement(predictions, labels)

            tp += sub_tp
            tn += sub_tn
            fp += sub_fp
            fn += sub_fn

    # Average loss
    val_loss /= len(val_loader.dataset)

    # Metrics
    val_acc = (tp + tn) / (tp + tn + fp + fn) * 100

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    f1_score = (2 * tp) / (2 * tp + fp + fn) \
        if (2 * tp + fp + fn) > 0 else 0

    print(
        f'↳ Val Loss: {val_loss:.4f}, '
        f'Val Acc: {val_acc:.2f}%'
    )

    print(
        f'↳ Recall: {recall:.4f}, '
        f'Precision: {precision:.4f}, '
        f'F1-score: {f1_score:.4f}'
    )

    return val_loss, val_acc, recall, precision, f1_score

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    parser = ArgumentParser()

    # Parser for model
    parser.add_argument('--num_classes', type=int, required=False, default=2)

    # Parser for training
    parser.add_argument('--num_epochs', type=int, required=False, default=50)
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-2)

    # Parser for dataloader
    parser.add_argument('--dataset', type=str, required=False, default='chest_xray')

    # Parser for data augmentation
    parser.add_argument('--degree', type=int, default=10)
    parser.add_argument('--resize', type=int, default=224)
    parser.add_argument('--scale', type=float, default=0.1)
    parser.add_argument('--contrast', type=float, default=0.3)
    parser.add_argument('--brightness', type=float, default=0.0)

    # Parser for switch the model
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--weight', type=str, default='pretrain')

    args = parser.parse_args()

    # set gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'## Now using {device} as calculating device ##')

    # set dataloader
    train_loader, test_loader, val_loader = CustomDataLoader.customDataLoader_ResNet_Basic(
        resize=args.resize,
        batch_size=args.batch_size,
        aug_Brightness=args.brightness,
        aug_Rotation=args.degree,
        aug_Contrast=args.contrast,
        aug_Scaling=args.scale
    )

    # define model 
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    if(args.model == 'resnet18'):
        if(args.weight == 'pretrain'):
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            model = models.resnet18()

    elif(args.model == 'resnet50'):
        if(args.weight == 'pretrain'):
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            model = models.resnet50()

    elif(args.model == 'resnet101'):
        if(args.weight == 'pretrain'):
            model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        else:
            model = models.resnet101()
    
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, args.num_classes)
    model = model.to(device)

    # define loss function, optimizer
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([3.8896346, 1.346]))
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # training
    print("=== Start Training ===")
    best_acc, best_f1_score, history = train(device, train_loader, val_loader, model, criterion, optimizer)

    # plot
    print("=== Train Result ===")
    plot_accuracy(history)
    plot_f1_score(history)
    print("Best acc:", best_acc)
    print("Best f1 score:", best_f1_score)

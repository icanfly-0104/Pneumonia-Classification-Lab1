import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torchvision.models as models
import warnings
from argparse import ArgumentParser
import CustomDataLoader

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

def test(test_loader, model, device):
    tp, tn, fp, fn = 0, 0, 0, 0

    with torch.no_grad():
        model.eval()

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            outputs = torch.max(outputs, 1).indices

            sub_tp, sub_tn, sub_fp, sub_fn = measurement(outputs, labels)

            tp += sub_tp
            tn += sub_tn
            fp += sub_fp
            fn += sub_fn

    # Standard confusion matrix
    c_matrix = [
        [int(tn), int(fp)],
        [int(fn), int(tp)]
    ]

    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_score = (2 * tp) / (2 * tp + fp + fn)

    return accuracy, f1_score, recall, precision, c_matrix

def test_and_plot(test_loader, model, device):
    # testing
    test_acc, f1_score, recall, precision, best_c_matrix = test(test_loader, model, device)

    print("===Testing Result===")
    print(f'Recall: {recall:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'F1-score: {f1_score:.4f}')
    print(f'Test Acc.(%): {test_acc:.2f}%')
    plot_confusion_matrix(best_c_matrix)

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    parser = ArgumentParser()

    # Parser for change the model
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--weight', type=str, default='pretrain')
    parser.add_argument('--path', type=str, default='model_weights.pt')
    
    args = parser.parse_args()

    # set gpu
    print("=== Start Testing ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'## Now using {device} as calculating device ##')

    #Load model weights
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

    model.fc = nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(torch.load(args.path,map_location=device))

    model = model.to(device)

    _, test_loader, _ = CustomDataLoader.customDataLoader_ResNet_Basic()
    test_and_plot(test_loader, model, device)
        
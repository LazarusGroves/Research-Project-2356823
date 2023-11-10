import torch
import os, timm
from timm import create_model
from timm.loss import LabelSmoothingCrossEntropy
from common import load_dataset, DATASETS
from sklearn.metrics import precision_score, recall_score, f1_score

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for datasetName in DATASETS:
        CEL = torch.nn.CrossEntropyLoss()
        CELNAME = 'CrossEntropyLoss'
        LSCEL = LabelSmoothingCrossEntropy()
        LSCELNAME = 'LabelSmoothingCrossEntropy'
            
        (train_loader, test_loader, val_loader), (train_len, test_len, val_len), num_classes, classes = load_dataset(datasetName)
        for modelName in os.listdir('Models/SwinT/{}'.format(datasetName)):
            #if modelName doesnt contain 'model_25' skip
            if 'model_25' not in modelName:
                continue
            model = create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=num_classes)
            model.load_state_dict(torch.load('Models/SwinT/{}/{}'.format(datasetName, modelName)))
            model.to(device)
            model.eval()
            correct = 0
            total = 0
            CEL_Loss = 0
            LSCEL_Loss = 0
            all_labels = []
            all_predicted = []

            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    images = images.cuda()
                    labels = labels.cuda()
                    outputs = model(images)

                    CEL_Loss += CEL(outputs, labels).item()
                    LSCEL_Loss += LSCEL(outputs, labels).item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    # Collect true labels and predicted labels for later use
                    all_labels.extend(labels.cpu().numpy())
                    all_predicted.extend(predicted.cpu().numpy())

            CEL_Loss /= len(test_loader)
            LSCEL_Loss /= len(test_loader)
            avg_acc = correct / total

            # Calculate precision, recall, and F1 Score
            precision = precision_score(all_labels, all_predicted, average='macro')
            recall = recall_score(all_labels, all_predicted, average='macro')
            f1 = f1_score(all_labels, all_predicted, average='macro')

            # Model Name, CEL Loss, LSCEL Loss, Precision, Recall, F1 Score, Accuracy
            print("Model Name,\tCEL Loss,\tLSCEL Loss,\tPrecision,\tRecall,\tF1 Score,\tAccuracy")
            print("{} {} {} {} {} {} {}".format(modelName, CEL_Loss, LSCEL_Loss, precision, recall, f1, avg_acc))
            del model
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
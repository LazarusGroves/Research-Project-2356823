import torch
import os
from timm import create_model
from common import load_dataset, DATASETS

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load every model in directory Models/ConvNeXt/datasetName
    # and print the accuracy of each model on the test set.
    for datasetName in DATASETS:
        (train_loader, test_loader, val_loader), (train_len, test_len, val_len), num_classes, classes = load_dataset(datasetName)
        for modelName in os.listdir('Models/ConvNeXt/{}'.format(datasetName)):
            model = create_model('convnext_base', pretrained=True, num_classes=num_classes)
            model.load_state_dict(torch.load('Models/ConvNeXt/{}/{}'.format(datasetName, modelName)))
            model.to(device)
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    images = images.cuda()
                    labels = labels.cuda()
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            print("Model: {}, Accuracy: {}".format(modelName, correct / total))
            del model
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()


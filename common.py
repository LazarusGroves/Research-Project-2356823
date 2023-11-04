"""
    THIS FILE IS NOT INTENDED TO BE RUN, HOWEVER YOU CAN RUN IT TO SEE THE SPLIT OF THE DATASETS
    ALONG WITH CLASS NAMES AND NUMBER OF CLASSES. IT IS USED FOR COMMON FUNCTIONS AND LOADING DATASETS
"""

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from PIL import Image
from torchvision import transforms as T
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from tqdm import tqdm
import torchinfo, os, copy, time, warnings, torch

warnings.filterwarnings("ignore")

DATASETS = ["LANDUSE", "XRAY", "CIFAKE"]

def load_dataset(datasetName, BATCH_SIZE=128):
    # If datasetName is not in the list, then it will return error
    dsList = ["CIFAKE", "LANDUSE", "XRAY"]
    if datasetName not in dsList:
        raise ValueError("Dataset name is not in the list: CIFAKE, LANDUSE, XRAY")

    default_transform = T.Compose([ # Default transform, bilinear interpolation and resize to 224x224
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD), # ImageNet normalizing due to pre-trained model
    ])

    transform = None
    if datasetName != "LANDUSE": # LANDUSE dataset does not need augmentation, it is already augmented from the original dataset using only rotation and random crop
        transform = T.Compose([ # Data augmentation, random horizontal and vertical flip, random crop to 224x224, random erasing with p=0.1
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.Resize(256),
            T.RandomCrop(224),
            T.ToTensor(),
            T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD), # ImageNet normalizing due to pre-trained model
            T.RandomErasing(p=0.1, value='random')
        ])
    else:
        transform = default_transform # If LANDUSE dataset, then use default transform

    dir = os.path.join('Datasets/')

    test_dataset = None
    train_dataset = None
    # Boiler plate match case 

    classes = None
    classesLen = None

    match datasetName:
        case "CIFAKE":
            train_dir = os.path.join(dir, 'CIFAKE/train/')
            test_dir = os.path.join(dir, 'CIFAKE/test/')

            train_dataset = ImageFolder(train_dir, transform=transform)
            classes = train_dataset.classes
            classesLen = len(train_dataset.classes)

            test_dataset = ImageFolder(test_dir, transform=default_transform)
            # No Validation data for CIFAKE so sample 20% of the test data for validation
            test_dataset, val_dataset = random_split(test_dataset, [int(len(test_dataset)*0.30), int(len(test_dataset)*0.70)])
            # Since this dataset is so big, we will take only 10% of each dataset
            train_dataset, _ = random_split(train_dataset, [int(len(train_dataset)*0.10), int(len(train_dataset)*0.90)])
            test_dataset, _ = random_split(test_dataset, [int(len(test_dataset)*0.10), int(len(test_dataset)*0.90)])
            val_dataset, _ = random_split(val_dataset, [int(len(val_dataset)*0.10), int(len(val_dataset)*0.90)])
        case "LANDUSE":
            train_dir = os.path.join(dir, 'UC_Merced_Land_Use/train/')
            test_dir = os.path.join(dir, 'UC_Merced_Land_Use/test/')
            val_dir = os.path.join(dir, 'UC_Merced_Land_Use/val/')
            train_dataset = ImageFolder(train_dir, transform=transform)
            classes = train_dataset.classes
            classesLen = len(train_dataset.classes)
            test_dataset = ImageFolder(test_dir, transform=default_transform)
            val_dataset = ImageFolder(val_dir, transform=default_transform)
        case "XRAY":
            train_dir = os.path.join(dir, 'chest-xray-classification/train/')
            test_dir = os.path.join(dir, 'chest-xray-classification/test/')
            val_dir = os.path.join(dir, 'chest-xray-classification/val/')
            train_dataset = ImageFolder(train_dir, transform=transform)
            classes = train_dataset.classes
            classesLen = len(train_dataset.classes)
            test_dataset = ImageFolder(test_dir, transform=default_transform)
            val_dataset = ImageFolder(val_dir, transform=default_transform)

    print("=========================================")
    print("Dataset: {}".format(datasetName))
    print("Train dataset size: {}".format(len(train_dataset)))
    print("Test dataset size: {}".format(len(test_dataset)))
    print("Validation dataset size: {}".format(len(val_dataset)))
    print("Number of classes: {}".format(len(classes)))
    print("Classes: {}".format(classes))
    print("=========================================")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Return:
    #   (train_loader, test_loader, val_loader): tuple of loaders
    #   (len(train_dataset), len(test_dataset), len(val_dataset)): tuple of dataset sizes
    #   len(train_dataset.classes): number of classes
    #   train_dataset.classes: list of class names
    return (train_loader, test_loader, val_loader), (len(train_dataset), len(test_dataset), len(val_dataset)), classesLen, classes

def add_head(model):
    for name, param in model.named_parameters():
        if 'head' not in name:
            param.requires_grad = False

    # torchinfo.summary(model, input_size=(128, 3, 224, 224), verbose=1)
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters: {}".format(trainable_params))
    return model

def train_model(
        model, 
        data_loaders, 
        dataset_sizes, 
        device, 
        criterion, 
        optimizer, 
        scheduler, 
        epochs=10,
        ):
    since = time.time()
    fallback_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    model = add_head(model)

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('====================')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_corrects = 0.0, 0.0

            for inputs, labels in tqdm(data_loaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'): # No autograd makes validation quicker
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = (running_corrects.double() / dataset_sizes[phase]).cpu().numpy()

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            if phase == 'val':
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print('') # Spacer

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, train_loss, train_acc, val_loss, val_acc

def graph_saver(
        modelName, 
        datasetName, 
        epochs, 
        learning_rate, 
        batch_size, 
        train_loss, 
        train_acc, 
        val_loss, 
        val_acc,
        ):
    import matplotlib.pyplot as plt
    import numpy as np

    # Make dir for dataset if it does not exist
    if not os.path.exists("Graphs/{}".format(datasetName)):
        os.makedirs("Graphs/{}".format(datasetName))

    # Save as two seperate svg graphs
    plt.figure()
    plt.plot(np.arange(len(train_loss)), train_loss, label="Train Loss")
    plt.plot(np.arange(len(val_loss)), val_loss, label="Validation Loss")
    plt.legend()
    plt.suptitle("{} Loss across {}".format(modelName, datasetName))
    plt.title('Epochs: {}, Learning Rate: {}, Batch Size: {}'.format(epochs, learning_rate, batch_size))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    for i in range(len(train_loss)):
        plt.annotate("{:.2f}".format(train_loss[i]), (i, train_loss[i]))
    for i in range(len(val_loss)):
        plt.annotate("{:.2f}".format(val_loss[i]), (i, val_loss[i]))
    plt.savefig("Graphs/{}/{}_{}_{}_{}_loss.svg".format(datasetName, modelName, epochs, learning_rate, batch_size))

    plt.figure()
    plt.plot(np.arange(len(train_acc)), train_acc, label="Train Accuracy")
    plt.plot(np.arange(len(val_acc)), val_acc, label="Validation Accuracy")
    plt.legend()
    plt.suptitle("{} Accuracy across {}".format(modelName, datasetName))
    plt.title('Epochs: {}, Learning Rate: {}, Batch Size: {}'.format(epochs, learning_rate, batch_size))
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    for i in range(len(train_acc)):
        plt.annotate("{:.2f}".format(train_acc[i]), (i, train_acc[i]))
    for i in range(len(val_acc)):
        plt.annotate("{:.2f}".format(val_acc[i]), (i, val_acc[i]))
    plt.savefig("Graphs/{}/{}_{}_{}_{}_acc.svg".format(datasetName, modelName, epochs, learning_rate, batch_size))

    plt.close('all')

if __name__ == '__main__':
    load_dataset("CIFAKE")
    load_dataset("LANDUSE")
    load_dataset("XRAY")
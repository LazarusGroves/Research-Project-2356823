from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from PIL import Image
from torchvision import transforms as T
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import os

def load_dataset(datasetName, BATCH_SIZE=64):
    # If datasetName is not in the list, then it will return error
    dsList = ["CIFAKE", "LANDUSE", "XRAY"]
    if datasetName not in dsList:
        raise ValueError("Dataset name is not in the list: CIFAKE, LANDUSE, XRAY")

    default_transform = T.Compose([ # Default transform, bilinear interpolation and resize to 224x224
        T.Resize(224),
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
    match datasetName:
        case "CIFAKE":
            train_dir = os.path.join(dir, 'CIFAKE/train/')
            test_dir = os.path.join(dir, 'CIFAKE/test/')
            train_dataset = ImageFolder(train_dir, transform=transform)
            test_dataset = ImageFolder(test_dir, transform=default_transform)
            # No Validation data for CIFAKE so sample 20% of the test data for validation
            test_dataset, val_dataset = random_split(test_dataset, [int(len(test_dataset)*0.30), int(len(test_dataset)*0.70)])
        case "LANDUSE":
            train_dir = os.path.join(dir, 'UC_Merced_Land_Use/train/')
            test_dir = os.path.join(dir, 'UC_Merced_Land_Use/test/')
            val_dir = os.path.join(dir, 'UC_Merced_Land_Use/val/')
            train_dataset = ImageFolder(train_dir, transform=transform)
            test_dataset = ImageFolder(test_dir, transform=default_transform)
            val_dataset = ImageFolder(val_dir, transform=default_transform)
        case "XRAY":
            train_dir = os.path.join(dir, 'chest-xray-classification/train/')
            test_dir = os.path.join(dir, 'chest-xray-classification/test/')
            val_dir = os.path.join(dir, 'chest-xray-classification/val/')
            train_dataset = ImageFolder(train_dir, transform=transform)
            test_dataset = ImageFolder(test_dir, transform=default_transform)
            val_dataset = ImageFolder(val_dir, transform=default_transform)

    print("=========================================")
    print("Dataset: {}".format(datasetName))
    print("Train dataset size: {}".format(len(train_dataset)))
    print("Test dataset size: {}".format(len(test_dataset)))
    print("Validation dataset size: {}".format(len(val_dataset)))
    print("=========================================")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=int(BATCH_SIZE/4), shuffle=False, num_workers=4, pin_memory=True)

    return (train_loader, test_loader, val_loader), (len(train_dataset), len(test_dataset), len(val_dataset))


load_dataset("CIFAKE")
load_dataset("LANDUSE")
load_dataset("XRAY")
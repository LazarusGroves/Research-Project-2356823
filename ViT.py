"""
    ViT PRETRAINED-22K MODEL
"""

from common import train_model, add_head, load_dataset, graph_saver, DATASETS
from timm.loss import LabelSmoothingCrossEntropy
from torch.optim import AdamW, lr_scheduler
import argparse, timm, torch, os, time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(EPOCHS, LEARNING_RATE, BATCH_SIZE):
    print("ViT Pretrained-22k Model")
    print("Epochs: {}, Learning Rate: {}, Batch Size: {}".format(EPOCHS, LEARNING_RATE, BATCH_SIZE))
    print("=========================================")
    for datasetName in DATASETS:
        (train_loader, test_loader, val_loader), (train_len, test_len, val_len), num_classes, classes = load_dataset(datasetName, BATCH_SIZE)
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)

        data_loaders = {
            "train": train_loader,
            "val": val_loader
        }
        dataset_sizes = {
            "train": train_len,
            "val": val_len
        }

        model = model.to(device)

        criterion = LabelSmoothingCrossEntropy()
        criterion = criterion.to(device)

        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        model, train_loss, train_acc, val_loss, val_acc = train_model(model, 
                                                                      data_loaders, 
                                                                      dataset_sizes, 
                                                                      device, 
                                                                      criterion, 
                                                                      optimizer, 
                                                                      exp_lr_scheduler, 
                                                                      EPOCHS)
            
        # Write the above portion to a file
        with open('ViT.txt', 'a') as f:
            f.write("=========================================\n")
            f.write("Model: ViT, Dataset: {}\n".format(datasetName))
            f.write("Epochs: {}, Learning Rate: {}, Batch Size: {}\n".format(EPOCHS, LEARNING_RATE, BATCH_SIZE))
            for i in range(EPOCHS):
                f.write("{:.4f} ".format(train_loss[i]))
            f.write("\n")
            for i in range(EPOCHS):
                f.write("{:.4f} ".format(train_acc[i]))
            f.write("\n")
            for i in range(EPOCHS):
                f.write("{:.4f} ".format(val_loss[i]))
            f.write("\n")
            for i in range(EPOCHS):
                f.write("{:.4f} ".format(val_acc[i]))
            f.write("\n")
            f.write("=========================================\n")

        graph_saver('ViT', datasetName, EPOCHS, LEARNING_RATE, BATCH_SIZE, train_loss, train_acc, val_loss, val_acc)

        # Create model directory for ViT if it doesn't exist
        if not os.path.exists('models/ViT'):
            os.makedirs('models/ViT')
        if not os.path.exists('models/ViT/{}'.format(datasetName)):
            os.makedirs('models/ViT/{}'.format(datasetName))

        torch.save(model.state_dict(), 'models/ViT/{}/model_{}_{}_{}.pt'.format(datasetName, EPOCHS, LEARNING_RATE, BATCH_SIZE))
        del model
        torch.cuda.empty_cache()

    return 0

if __name__ == '__main__':
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', type=int, default=10)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-b', type=int, default=128)

    args = parser.parse_args()

    main(args.e, args.lr, args.b)
    """

    # epochs = [5, 10]
    # learning_rates = [0.0001, 0.001]
    # batch_sizes = [64, 128]
    epochs = [25]
    learning_rates = [0.001]
    batch_sizes = [64]

    start_time = time.time()

    for e in epochs:
        for lr in learning_rates:
            for b in batch_sizes:
                main(e, lr, b)


    end_time = time.time()
    print("Total time taken: {}".format(end_time - start_time))

    # Write to file
    with open('ViT.txt', 'a') as f:
        f.write("Total time taken: {}".format(end_time - start_time))
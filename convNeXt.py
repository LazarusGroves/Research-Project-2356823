"""
    CONVNEXT PRETRAINED-22K MODEL
"""

from common import train_model, add_head, load_dataset, graph_saver, DATASETS
import argparse

def main(EPOCHS, LEARNING_RATE, BATCH_SIZE):
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', type=int, default=10)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-b', type=int, default=128)

    args = parser.parse_args()

    main(args.e, args.lr, args.b)
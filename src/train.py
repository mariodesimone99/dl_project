import sys
import yaml
import torch
from torch.utils.data import DataLoader
from utils import count_params
from cityscapes_dataset import CityscapesDataset
from cross_stitchnet import CrossStitchNet
from densenet import DenseNet
from depthnet import DepthNet
from mtan import MTAN
from segnet import SegNet
from splitnet import SplitNet
from stan import STAN
from trainer import Trainer

#TODO. prepare config files for each dataset and labels

BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 0.0001
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DATASET_PATH = "./dataset/cityscapes_preprocessed"
LABELS = 7

if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)

    print(f"Using device: {DEVICE}")
    print(f"Training for {EPOCHS} epochs")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}\n")

    model_name = config["model_name"]

    if model_name == "cross_stitch":
        model = CrossStitchNet(filter=config['filter'], classes=config['classes'])
    elif model_name == "densenet":
        model = DenseNet(filter=config['filter'], classes=config['classes'])
    elif model_name == "depthnet":
        model = DepthNet(filter=config['filter'], mid_layers=config['mid_layers'])
    elif model_name == "mtan":
        model = MTAN(filter=config['filter'], classes=config['classes'])
    elif model_name == "segnet":
        model = SegNet(filter=config['filter'], classes=config['classes'], mid_layers=config['mid_layers'])
    elif model_name == "splitnet":
        model = SplitNet(filter=config['filter'], classes=config['classes'], mid_layers=config['mid_layers'])
    elif model_name == "stan_dep" or model_name == "stan_seg":
        model = STAN(filter=config['filter'], classes=config['classes'])
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    print(f"Model will be trained on CityScapes Dataset with {LABELS} labels")
    train_dataset = CityscapesDataset(root=DATASET_PATH, split='train', labels=LABELS)
    val_dataset = CityscapesDataset(root=DATASET_PATH, split='val', labels=LABELS)
    
    train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Model chosen: {model_name}")
    print(f"{model.name} has {count_params(model)} parameters\n")

    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    trainer = Trainer(model, opt)
    print("------------------Starting Training------------------")
    trainer.train(train_dl, val_dl=val_dl, epochs=EPOCHS, save=True, grad=True)
    print("------------------Training complete------------------")

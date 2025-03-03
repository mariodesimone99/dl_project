import sys
import yaml
import torch
from torch.utils.data import DataLoader
from utils import count_params
from cityscapes_dataset import CityscapesDataset
from nyuv2_dataset import NYUv2Dataset
from cross_stitchnet import CrossStitchNet
from densenet import DenseNet
from depthnet import DepthNet
from mtan import MTAN
from normalnet import NormalNet
from segnet import SegNet
from splitnet import SplitNet
from stan import STAN
from trainer import Trainer

#TODO: prepare config files for each dataset and labels



# BATCH_SIZE = 8
# EPOCHS = 100
# LEARNING_RATE = 0.0001
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# DATASET_PATH = "./dataset/cityscapes_preprocessed"
# LABELS = 7

def instance_model(config):
    models_dict = {}
    models_dict['cross_stitch'] = CrossStitchNet(filter=config['filter'], classes=config['classes'], tasks=config['tasks'], depth_activation=config['depth_activation'])
    models_dict['densenet'] = DenseNet(filter=config['filter'], classes=config['classes'], mid_layers=config['mid_layers'], tasks=config['tasks'], depth_activation=config['depth_activation'])
    models_dict['depthnet'] = DepthNet(filter=config['filter'], mid_layers=config['mid_layers'], depth_activation=config['depth_activation'])
    models_dict['mtan'] = MTAN(filter=config['filter'], mid_layers=config['mid_layers'], classes=config['classes'], tasks=config['tasks'], depth_activation=config['depth_activation'])
    models_dict['normalnet'] = NormalNet(filter=config['filter'], mid_layers=config['mid_layers'])
    models_dict['segnet'] = SegNet(filter=config['filter'], mid_layers=config['mid_layers'], classes=config['classes'])
    models_dict['splitnet'] = SplitNet(filter=config['filter'], mid_layers=config['mid_layers'], classes=config['classes'], tasks=config['tasks'], depth_activation=config['depth_activation'])
    models_dict['stan'] = STAN(filter=config['filter'], mid_layers=config['mid_layers'], classes=config['classes'], tasks=config['tasks'], depth_activation=config['depth_activation'])
    if config['model_name'] not in models_dict.keys():
        raise ValueError(f"Model {config['model_name']} not supported")
    
    return models_dict[config['model_name']]

def instance_dataset(config):
    splits = ['train', 'val']
    dataset_dict = {'cityscapes': {}, 'nyuv2': {}}
    if config['dataset_name'] not in dataset_dict.keys():
        raise ValueError(f"Dataset {config['dataset_name']} not supported")
    for split in splits:
        dataset_dict['cityscapes'][split] = CityscapesDataset(root=config['dataset_path'], split=split, labels=config['classes'])
    for split in splits:
        dataset_dict['nyuv2'][split] = NYUv2Dataset(root=config['dataset_path'], split=split)
    return dataset_dict[config['dataset_name']]['train'], dataset_dict[config['dataset_name']]['val']

if __name__ == "__main__":
    config = {}
    for arg in sys.argv[1:]:
        config_tmp = yaml.safe_load(open(arg, "r"))
        config = {**config, **config_tmp}
    # with open(sys.argv[1], "r") as f:
    #     config = yaml.safe_load(f)

    print(f"Using device: {DEVICE}")
    print(f"Training for {config['epochs']} epochs")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['lr']}\n")

    model = instance_model(config)

    # model_name = config["model_name"]

    # if model_name == "cross_stitch":
    #     model = CrossStitchNet(filter=config['filter'], classes=config['classes'])
    # elif model_name == "densenet":
    #     model = DenseNet(filter=config['filter'], classes=config['classes'])
    # elif model_name == "depthnet":
    #     model = DepthNet(filter=config['filter'], mid_layers=config['mid_layers'])
    # elif model_name == "mtan":
    #     model = MTAN(filter=config['filter'], classes=config['classes'])
    # elif model_name == "segnet":
    #     model = SegNet(filter=config['filter'], classes=config['classes'], mid_layers=config['mid_layers'])
    # elif model_name == "splitnet":
    #     model = SplitNet(filter=config['filter'], classes=config['classes'], mid_layers=config['mid_layers'])
    # elif model_name == "stan_dep" or model_name == "stan_seg":
    #     model = STAN(filter=config['filter'], classes=config['classes'])
    # else:
    #     raise ValueError(f"Model {model_name} not supported")
    
    print(f"Model will be trained on {config['dataset_name']} Dataset with {config['classes']} labels")
    # train_dataset = CityscapesDataset(root=config['dataset_path'], split='train', labels=config['classes'])
    # val_dataset = CityscapesDataset(root=config['dataset_path'], split='val', labels=config['classes'])
    train_dataset, val_dataset = instance_dataset(config)
    
    train_dl = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False)
    val_dl = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    print(f"Model chosen: {model.name}")
    print(f"{model.name} has {count_params(model)} parameters\n")

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    trainer = Trainer(model, opt, config['dataset_name'], DEVICE)
    print("------------------Starting Training------------------")
    trainer.train(train_dl, val_dl=val_dl, epochs=config['epochs'], save=True, grad=True)
    print("------------------Training complete------------------")

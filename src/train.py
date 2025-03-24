import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import count_params, visualize_results
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

# EPOCHS = 100
# LEARNING_RATE = 1e-4
# DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

def instance_model(config):
    filter = config['filter']
    models_dict = {}
    models_dict['cross_stitch'] = CrossStitchNet(filter=filter, classes=config['classes'], tasks=config['tasks'], depth_activation=config['depth_activation'])
    models_dict['densenet'] = DenseNet(filter=filter, classes=config['classes'], mid_layers=config['mid_layers'], tasks=config['tasks'], depth_activation=config['depth_activation'])
    models_dict['depthnet'] = DepthNet(filter=filter, mid_layers=config['mid_layers'], depth_activation=config['depth_activation'])
    models_dict['mtan'] = MTAN(filter=filter, mid_layers=config['mid_layers'], classes=config['classes'], tasks=config['tasks'], depth_activation=config['depth_activation'])
    models_dict['normalnet'] = NormalNet(filter=filter, mid_layers=config['mid_layers'])
    models_dict['segnet'] = SegNet(filter=filter, mid_layers=config['mid_layers'], classes=config['classes'])
    models_dict['splitnet'] = SplitNet(filter=filter, mid_layers=config['mid_layers'], classes=config['classes'], tasks=config['tasks'], depth_activation=config['depth_activation'])
    models_dict['stan'] = STAN(filter=filter, mid_layers=config['mid_layers'], classes=config['classes'], task=config['tasks'][0], depth_activation=config['depth_activation'])
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
    # config = {}
    # for arg in sys.argv[1:]:
    #     config_tmp = yaml.safe_load(open(arg, "r"))
    #     config = {**config, **config_tmp}

    # config_model = yaml.safe_load(open(sys.argv[1], "r"))
    # config_dataset = yaml.safe_load(open(sys.argv[2], "r"))
    config_model = yaml.safe_load(open("./config/prova.yaml", "r"))
    config_dataset = yaml.safe_load(open("./config/prova_dataset.yaml", "r"))
    config = {**config_dataset, **config_model}
    config['depth_activation'] = nn.ReLU() if config['depth_activation'] == 'relu' else nn.Sigmoid()
    config['dwa'] = False
    # with open(sys.argv[1], "r") as f:
    #     config = yaml.safe_load(f)

    EPOCHS = config['epochs']
    BATCH_SIZE = config['batch_size']
    LEARNING_RATE = config['lr']
    DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
    if len(config['tasks']) > 1:
        ok = False 
        while ok:
            w_scheme = input(f'Dynamic Weight Average (y/n)?: ')
            if w_scheme == "y":
                config['dwa'] = True
                ok = True
            elif w_scheme == "n":
                ok = True
            else:
                continue


    print(f"Using device: {DEVICE}")
    print(f"Training for {EPOCHS} epochs")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}\n")

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
    
    print(f"Model will be trained on {config['dataset_name']} Dataset for tasks: {config['tasks']}")
    # train_dataset = CityscapesDataset(root=config['dataset_path'], split='train', labels=config['classes'])
    # val_dataset = CityscapesDataset(root=config['dataset_path'], split='val', labels=config['classes'])
    train_dataset, val_dataset = instance_dataset(config)
    
    train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Model chosen: {config['model_name']}")
    print(f"{model.name} has {count_params(model)} parameters\n")

    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    trainer = Trainer(model, opt, config['dataset_name'], DEVICE, dwa=config['dwa'])
    print("------------------Starting Training------------------")
    trainer.train(train_dl, val_dl=val_dl, epochs=EPOCHS, save=True, grad=True)
    print("------------------Training complete------------------")

    id_result = 0
    nresults = 10
    for i, (image, out) in enumerate(val_dl):
        state = visualize_results(model, DEVICE, image, out, id_result)
        id_result += BATCH_SIZE
        if state:
            break

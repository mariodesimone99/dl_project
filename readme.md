# Multi-Task Learning Models
This project is an implementation of the architectures [Cross-Stitch Network](https://arxiv.org/pdf/1604.03539) and [Multi-Task Attention Network](https://arxiv.org/pdf/1803.10704), to perform comparison between single-task models and multi-task ones.
The experiments have been made trying to reproduce results obtained in the MTAN paper, so models have been trained on Cityscapes and NYUv2 datasets (see section [Datasets](#datasets) for details)

## Index

- [Project Structure](#project-structure)
- [How to Start](#how-to-start)
- [Datasets](#datasets)
- [Code Comments](#code-comments)

## Project Structure

- **`config/`**: Contains the `.yaml` files with the experiments settings.
- **`scripts/`**: Contains the `.sh` files to run training experiments.
- **`dataset/`**: Must contain the preprocessed dataset (see section [Datasets](#datasets) for details).
- **`results/`**: Contains the images results for the evaluation for each dataset.
- **`models/`**: Will contain the `.pth` file of the models, saved during the training and the figure plot of statistics and loss.
- **`runs/`**: Will contain the tensorboard files to compare the models.
- **`src/`**: Source code of models, utils, datasets and the demo notebook.

Every folder with the exception of **`src/`** and **`config/`** has a subfolder for each dataset to isolate the experiments, models weights etc...

## How to start

In the root directory there is a file `requirements.txt`, conatining the list of all the packages used in the project, using the line
`pip install -r requirements.txt`
Once installed the requirements, choose a dataset and navigate through the relative `scripts` folder (for example for cityscapes):
`cd scripts/cityscapes`
then choose a model and execute (for example mtan):
`bash mtan_train.sh`
for multi-task models the terminal will ask if it should perform Dynamic Weight Averaging or instead use equal weights scheme.
Change the model architecture is easy, just navigate to its config folder, in the previous example:
`cd config/mtan.yaml`
To add a task it is necessary to go to the corresponding model source code, and add a task dependant head, in the example:
`cd src/mtan.py`
Finally the `demo.ipynb` notebook, has everything necessary to perform the training and the evaluation of a model in the interactive environment of a jupyter notebook

## Datasets
For the experiments have been used preprocessed (by the author of the MTAN paper):
- [NYUv2](https://www.dropbox.com/scl/fo/p7n54hqfpfyc6fe6n62qk/AKVb28ZmgDiGdRMNkX5WJvo?rlkey=hcf31bdrezqjih36oi8usjait&e=1&dl=0): dataset for segmentation, depth and normal estimation, [288x384] images of indoor scenes

- [Cityscapes](https://www.dropbox.com/scl/fo/x2i67p14fxy3d3178i8ln/AGZHvvk82ayNbcBHp-N0JXg?rlkey=ebhrpay2cgqnr0ew5vmti2diy&e=1&dl=0): dataset for segmentation and depth estimation of outdoor scenes

# Code Comments

The code has an extensive use of dictionaries to make abstraction of the task dependence, this let a loose coupling between task, training, statistics computation and model definition, to add a task it is necessary to implement its head in the model, define the statistics and add to the training pipeline.
# Thalia

This is the official repository for `Thalia`, a global dataset for volcanic activity monitoring through InSAR imagery. `Thalia` builds upon `Hephaestus` ([Bountos et al., 2022](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/html/Bountos_Hephaestus_A_Large_Scale_Multitask_Dataset_Towards_InSAR_Understanding_CVPRW_2022_paper.html)) and offers various key advantages:

 - Machine Learning-ready state
 - Georeferenced InSAR imagery (vs. PNG format)
 - Enhanced spatial resolution at 100m GSD (vs. 333m GSD)
 - Physically interpretable pixel values (vs. RGB composites)
 - Zarr file format with spatial and temporal dimensions
 - Additional data critical for volcanic monitoring: Digital Elevation Model (DEM), Confounding stmospheric variables

Annotations offer rich insights on the **deformation type** (sill, dyke, mogi, spheroid, earthquake), the **intensity level** (low, medium, high), the presence of **non-volcanic related fringes** (atmospheric, orbital, glacier) as well as the **volcanic activity phase** (rest, unrest, rebound). Each sample also contains a **text description** of the observed phenomena, ideal for language-based machine learning modelling.

You can explore a sample minicube from the dataset and investigate its structure and available annotation variables using the interactive Google Colab notebook below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NeVtXEqrAawe0ICw1prMJSuFdHPHOqlg)

## Download the dataset
You can either download the full dataset and produce your own dataset (option 1) or use the already implemented splits via huggingface (option 2).

Option 1:

1. **Download the full dataset**

   The latest version of the dataset is available here: [Thalia_v0](https://www.dropbox.com/scl/fo/o3dlvfs5d0uqh7fm3clqe/ALR3vXd40SyOIxzwwOC5_po?rlkey=ery6e44t5u0osgryyt1qq9z4b&st=kyi2lmek&dl=0)

2. **Export webdatasets**

 You need to set the `webdataset` parameter to `true` in the configuration file. If a webdataset for the timeseries length specified in the configuration file does not exist, executing the `main.py` script will run on "webdataset creation mode". You need to run the same command 3 times (for each of train, val and test sets) in order to create the full webdataset. When that process ends, run `./webdataset_renaming.sh` (you will need to change the bash script to specify the base directory and the timeseries length to rename).

 Option 2:
 1. **Download the webdatasets**

The webdatasets explained in the paper and used for the benchmark are available via huggingface:


## Benchmark

The code in this repository implements an extensive benchmark on `Thalia` with a wide range of state of the art Deep Learning models. The benchmark consists of two basic tasks: **image classification** and **semantic segmentation**, each with both single-image and time-series input. Below we list the models used in the experimentation:

Classification:

- ResNet
- MobileNet v3
- EfficientNet v2
- ConvNeXt
- ViT

Segmentation:

- DeepLab v3
- UNet
- SegFormer

## Data split

We consider a temporal split, using data in `01/2014`-`05/2019` for training, `06/2019`-`12/2019` for validation and `01/2020`-`12/2021` for testing.

## How to train a model

To train and evaluate a Deep Learning model, simply run the following (with optional flags):

`python main.py`

Flags:

`--wandb` to sync results to a Wandb (Weights and Biases) project specified by configs

The model and backbone to use, as well as various training hyperparameters (e.g. batch size, number of epochs, learning rate, etc.) need to be configured in the configuration file (`configs/configs.json`).

## Example notebook

We provide a [Jupyter notebook](https://github.com/paren8esis/Hephaestus-minicubes/blob/main/HephaestusMinicubes_sample.ipynb) showing how to download and explore a minicube.

## Notes

### Timeseries implementation

We group per primary_date and get multiple secondary_dates --> p, [s1, ..., sn].
- If len([s1, ..., sn]) > `timeseries_length` we choose a random subset for every epoch
- If len([s1, ..., sn]) < `timeseries_length` we do random duplication every epoch so that we reach the desired length

### Undersampling
We export all available samples in the WebDataset format, but during training, we sample all positive examples and an equal number of negative examples per epoch.

### Cropping
We apply a 512x512 crop with a random offset from the frame center, ensuring that the deformation is included in the crop if present.

### Target mask
There are 4 main options for the creation of the target mask (defined as `mask_target` in the configuration file):

- `Last`: Use the mask of the last sample in the time-series
- `Peak`: Use a sum over all masks in the time-series
- `Union`: Use a union of all masks in the time-series
- `All`: Return all masks in the time-series (Only used for debugging)

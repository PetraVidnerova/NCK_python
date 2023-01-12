# Tools for evaluation of user defined networks

## Requirements 

pytorch, torchvision, pandas, click

## Data Preprocessing

1. First of all we expect to have a directory structure with subdirectories `obX`, where
 `X` is a number, containing JPG images in `camera_imgs` subfolder and coresponding YAML files (`processed_data.yaml`).
 
 such as
```
 exp7500/
   |
   -- ob1
       |
       -- camera_imgs/    (folder with JPG files)
       |
       -- processed_data.yaml
```


2. As a preliminary step, we create data list using `create_list.py`:   
   ```
   python create_list.py ../exp7500/
   ```
   This creates `data_list.csv`, this CSV file can be used for RobotDataSet (see `dataset.py`) or `resize.py` script.  
   
3. [OPTIONAL] You can preprocess images by `resize.py`. It creates mirror direcory containg `.pt` file
   for each original image. 
   ```
   Usage: resize.py [OPTIONS] SRC DEST

   Options:
     -x, --sizex INTEGER
     -y, --sizey INTEGER
     --help               Show this message and exit.

   ```
   For example:
   ```
   python resize.py -x 256 -y 192 ../exp7500/ ../data256x192
   ```
   
4. You can use `RobotDataSet` or `RobotPreprocessedDataset` (in case you have preprocessed images) for training (see `test.py` for example).



## Semi-automatic net search

A tool for simple evaluating of network architectures.

1. Create a YAML file with list of network architectures. See `networks.yaml` for the example of syntax. 

2. [OPTIONAL] Create a YAML file with configuration of training. See `train_cfg_example.yaml`. 

3. Run `train_para.py`.
```
Usage: train_para.py [OPTIONS] NETWORKS [TRAINCFG]

Options:
  --data-root TEXT
  --input_shape TEXT
  --help              Show this message and exit.
```
for example:
```
python train_para.py networks.yaml --data-root ../data256x192/ --input_shape "256,192"
```

## Postprocessing

Use the script `validity.py` to check the success rate (in the robot task) of your network.
```
python validity.py my_net.pt
```

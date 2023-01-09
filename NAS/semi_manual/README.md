# Tools for evaluation of user defined networks


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


2. As a preliminary step, we create data list using `data.py`:   
   ```
   python data.py ../exp7500/
   ```
   This creates `data_list.csv`, this CSV file can be used for RobotDataSet (see `dataset.py`) or `resize.py` script.  
   
2. preprocess images by `resize.py` (create mirror direcory containg `.pt` file
   for each original image)
   
3. use RobotPreprocessedDataset for training (see `test.py` for example)

## Semi-automatic net search

A tool for simple evaluating of network architectures.

# Tools for evaluation of user defined networks


## Data Preprocessing

1. create data list using `data.py` (outputs CSV file), 
   this CSV file can be used for RobotDataSet (see `dataset.py`).
   
2. preprocess images by `resize.py` (create mirror direcory containg `.pt` file
   for each original image)
   
3. use RobotPreprocessedDataset for training (see `test.py` for example)

## Semi-automatic net search

A tool for simple evaluating of network architectures.

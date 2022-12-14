# Data Preprocessing

1. create data list using `data.py` (outputs CSV file), 
   this CSV file can be used for RobotDataSet (see `dataset.py`).
   
2. preprocess images by `resize.py` (create mirror direcory containg `.pt` file
   for each original image)
   
3. use RobotPreprocessedDataset for training (see `test.py` for example)

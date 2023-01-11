# Auto-NAS

An  automatic architecture search tool based on multi-objective evolution. It optimises
both network perfomance and an architecture size.  

## Keywords:
deep neural networks, convolutional neural networks, automatic model selection, neural architecture search, multiobjective optimisatio, NSGA, NSGAII, NSGA3

## Requirements:

numpy, keras, pandas, scikit-learn, deap, click, matplotlib

## Main features:
- supports keras datasets, data form csv files, data from tensorflow records
- implements "vanilla" GA, multiobjective evolution via NSGA, NSGAII, NSGAIII 
- runs in parallel on one GPU or parallel on several CPUs 
- optimises feedworfard deep neural networks with dense layers, convolutinal networks   
 
## Usage:
1. Run evolution using `main.py`, produces a `.json` file with the list of all architectures from the pareto-front, a `.pkl` file with the checkpoint (after each iteration). Checkpoint stores all information
 needed to continue the computation in another run as well as the results. 
2. Inspect results runing `evaluate_result.py` on the resulting `.pkl` checkpoint file 

### main.py: 
```
usage: main.py [-h] [--id ID] [--log LOG] TASK

positional arguments:
  TASK        name of a yaml file with a task definition

optional arguments:
  -h, --help  show this help message and exit
  --id ID     computation id
  --log LOG   set logging level, default INFO
```

Example:
```
python main.py task.yaml --id test1
```
to run on one GPU (recommended), I use: 
```
CUDA_VISIBLE_DEVICES=0 python main.py tasks.yaml --id gpu_test 
```
### evaluate_result.py 
```
Usage: evaluate_result.py [OPTIONS] COMMAND [ARGS]...

Options:
  --conv BOOLEAN
  --help          Show this message and exit.

Commands:
  eval-front
  evaluate
  list-front
  plot
  query-iter
``` 

Example:
```
python evaluate_result.py eval-front  --data_source keras --trainset mnist checkpoint.pkl
```

## Config file example

task_nck.yaml 
```
dataset:
  name: gan_256192
  source_type: tfrecords
  ximg: 256
  yimg: 192
  len_train: 1963 
  
network_type: conv

nsga: 2

main_alg:
  batch_size: 8
  eval_batch_size: 1
  epochs: 60
  loss: mean_squared_error
  task_type: regression
  final_epochs: 100

ga:
  pop_size: 10
  n_gen: 50
  
network:
  max_layers: 5
  max_layer_size: 300
  min_layer_size: 5
  dropout: [0.0, 0.2, 0.3, 0.4]
  activations: ['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
  max_conv_layers: 4
  conv_layer: 0.7
  max_pool_layer: 0.3
  min_pool_size: 2
  max_pool_size: 4
  min_filters: 10
  max_filters: 50
  max_dense_layers: 4
  min_kernel_size: 2
  max_kernel_size: 5

device:
  device_type: GPU
```

To run on GPU specify:
```
device:
  device_type: GPU
``` 

If no GPU, use `device_type: CPU` and `n_cpus` that forces a use of multiprocessing with the given number of workers. 

To evolve convolutional networks:
```
network_type: conv
``` 

To use data from `csv` file (file should be comma separated, 
without a header, output variable in the last column):
```
dataset:
  source_type: csv
  name: data_train.csv
  test_name: data_test.csv
```

 

import sys 
from operator import mul
from functools import reduce

import numpy as np
import pandas as pd

import click

from tqdm import tqdm 
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.multiprocessing import set_start_method

from utils import train, evaluate
from config import read_config
from load_networks import load_network_configs, create_network
from data_robot import create_data_loader
from pool import Pool

PRETRAINED=False
BATCH_SIZE=64


def evaluate_network(net, train_cfg, random_seed=None, device="cpu"):
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    train_dl, val_dl, _ = create_data_loader(BATCH_SIZE,
                                             train_val=True,
                                             test=False)
        
    net = train(
        net,
        train_dl, val_dl, 
        **train_cfg,
        device=device
    )
    _, _, test_dl = create_data_loader(BATCH_SIZE, train_val=False, test=True)
    return evaluate(net, test_dl, train_cfg["criterion"], device)
    

def evaluate_task(task_config, device):
    netstr, train_cfg, input_shape, random_seed = task_config
    try:
        name, net =  create_network(netstr, input_shape, random_seed=random_seed)
        print(f"Going to evaluate network {name}")
        res = evaluate_network(net, train_cfg, random_seed=random_seed, device=device)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"Network {name} out of memory.")
            return name, None
        else:
            print(f"Network {name}: ", e)
            return name, None

    print(f"Evaluation of {name} finished. Saving ...")    
    torch.save(net, f"{name}.pt")
    print(f"{name}.pt saved.")
    return name, res
    
@click.command()
@click.argument('networks')
@click.argument('traincfg', default="train_cfg_example.yaml")
def test(networks, traincfg):

    
    train_cfg = read_config(traincfg)
    input_shape = (1, 512, 512)
    
    if torch.cuda.is_available():
        processors = torch.cuda.device_count()
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        devices = ["cpu"] * 2
        processors = 2

    set_start_method('spawn')
    pool = Pool(processors=processors, evalfunc=evaluate_task, devices=devices)

    results = []
    num = 0
    for cfg in load_network_configs(networks):
        
        task_config = (cfg, train_cfg, input_shape, 42)
        pool.putQuerry(task_config)
        num += 1

    for _ in range(num):
        name, res = pool.getAnswer()
        results.append({"network_name": name, "result": res})
    pool.close()
    print("EVALUATION FINISHED")
    df = pd.DataFrame(results)
    print(df)
    df.to_csv("results.csv")
    print("Results saved to results.csv")
    

if __name__ == "__main__":
    test()

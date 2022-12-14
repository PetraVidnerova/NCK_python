from math import floor
import yaml
import torch
import torch.nn  as nn

from utils import get_class_name

def i_or_f(x):
    try:
        return int(x)
    except:
        try:
            return float(x)
        except:
            raise ValueError("Should be int or float.")

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)
    
def create_network(cfg, input_shape, random_seed=None):
    if random_seed is not None:
        torch.manual_seed(random_seed)
        
    net = nn.Sequential()

    input_shape = input_shape
    if input_shape[1] != input_shape[2]:
        raise NotImplementedError
    
    for i, layer in enumerate(cfg["layers"]):
        print(f"Processing layer {i}")
        name = layer.split(";")[0]
        class_name = get_class_name("torch.nn", name)
        kwargs = { kw.split(":")[0].strip(): i_or_f(kw.split(":")[1].strip())
                  for kw in layer.split(";")[1:] }

        if name == "Conv2d":
            layer = class_name(input_shape[0], **kwargs)
            input_shape = (
                kwargs["out_channels"],
                floor((input_shape[1] - layer.kernel_size[0] + 2*layer.padding[0])/layer.stride[0])+1,
                floor((input_shape[1] - layer.kernel_size[0] + 2*layer.padding[0])/layer.stride[0])+1
            )
        elif name == "MaxPool2d":
            layer = class_name(**kwargs)
            input_shape = (
                input_shape[0],
                floor((input_shape[1] - layer.kernel_size + 2*layer.padding)/layer.stride)+1,
                floor((input_shape[1] - layer.kernel_size + 2*layer.padding)/layer.stride)+1
            )
        elif name == "Flatten":
            layer = Flatten()
            input_shape = input_shape[0] * input_shape[1] * input_shape[2]
        elif name == "Linear":
            layer = class_name(input_shape, **kwargs)
            input_shape  = kwargs["out_features"] 
        else:
            layer = class_name(**kwargs)
        net.append(layer)
    
    return cfg["name"], net


def load_network_configs(filename):

    with open(filename, "r") as f:
        text = f.read()

    network_configs = yaml.safe_load(text)

    for cfg in network_configs:
        yield cfg


def load_network(filename, input_shape, random_seed=None):

    for cfg in load_network_configs(filename):
        yield create_network(cfg, input_shape, random_seed)

import yaml
import torch
import torchvision


alexnet = torchvision.models.alexnet(pretrained=False)
print(alexnet)
exit()

list_of_layers = []
for layer in list(alexnet.features) + [alexnet.avgpool] + list(alexnet.classifier):
    name = layer.__class__.__name__
    args = {}
    if name == "Conv2d":
        print(dir(layer))
        layerstr = f"{name};out_channels: {layer.out_channels};kernel_size: {layer.kernel_size[0]};stride: {layer.stride[0]};padding: {layer.padding[0]}"
    elif name == "MaxPool2d":
        layerstr = f"{name};kernel_size: {layer.kernel_size};stride: {layer.stride};padding: {layer.padding}"
    elif name == "AdaptiveAvgPool2d":
        layerstr = f"{name};output_size: {layer.output_size[0]}"
    elif name == "Dropout":
        layerstr = f"{name};p: {layer.p}"
    elif name == "Linear":
        layerstr = f"{name};out_features: {layer.out_features}"
    else:
        layerstr = name
    list_of_layers.append(layerstr)


print(list_of_layers)

list_of_networks = [
    {"name": "alexnet", "layers": list_of_layers},
    {"name": "net2", "layers": list_of_layers.copy()}
]

with open("networks.yaml", "w") as f:
    f.write(yaml.dump(list_of_networks))

    

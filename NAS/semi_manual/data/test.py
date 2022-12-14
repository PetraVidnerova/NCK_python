import random
from tqdm import tqdm
import torch
import torchvision
from torch.nn import Conv2d, Linear
from torch.utils.data import DataLoader

from data.dataset import RobotPreprocessedDataSet

def create_network():
    alexnet = torchvision.models.alexnet(weights=None)

    # modify it to fit the task
    alexnet.features[0] = Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    
    alexnet.classifier[4] = Linear(in_features=4096, out_features=1024, bias=True)
    alexnet.classifier[6] = Linear(in_features=1024, out_features=6, bias=True)

    return alexnet

def evaluate(net, test_dl, device="cpu"):
    net.eval()
    loss_val = 0
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for data, targets in tqdm(test_dl):
            data = data.to(device=device)
            targets = targets.to(device=device)
            # Forward Pass
            out = net(data)
            loss_val += criterion(out, targets).item()
    print(f"VAL loss: {loss_val/len(test_dl)}")

def train(net,
          train_dl,
          val_dl, 
          epochs, 
          device="cpu"
):

    learning_rate = 1e-4

    net.to(device=device)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)    
    criterion = torch.nn.MSELoss()
    
    for epoch in range(epochs):
        loss_ep = 0

        print(f"Epoch {epoch} ... ")
        net.train()
        with tqdm(total=len(train_dl)) as t:
            for batch_idx, (data, targets) in enumerate(train_dl):
                data = data.to(device=device)
                targets = targets.to(device=device)
                ## Forward Pass
                optimizer.zero_grad()
                out = net(data)
                loss = criterion(out, targets)
                loss.backward()
                optimizer.step()
                loss_ep += loss.item()
                t.update()
            
        print(f"Loss in epoch {epoch} :::: {loss_ep/len(train_dl)}")

        evaluate(net, val_dl, device=device)
    return net





def create_data_loader(fname="data_list.csv", image_list=None, shuffle=True, data_root="./exp7500_512x512", transform=None, rotate=False):

    ds = RobotPreprocessedDataSet(fname, data_root=data_root, image_list=image_list, transform=transform, rotate=rotate)
    dl = DataLoader(ds, batch_size=32, shuffle=shuffle, num_workers=8)
    return dl


def main():
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    net = create_network()

    train_list = [
        1,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26,28,29,31,32,34,35,37,38,40,41,
        43,44,46,48,50,51,53,54,56,57,59,60,62,63,65,66,71,72
    ]
    test_list = [
        6,9,12,15,18,21,24,27,30,33,36,39,42,45,49,52,55,58,61,64,67,73
    ]

    random.shuffle(train_list)
    train_list, val_list = train_list[:-11], train_list[-11:]

    train_dl = create_data_loader(image_list=train_list)
    val_dl = create_data_loader(image_list=val_list)
    
    train(net, train_dl, val_dl, 100, device=device)

    test_dl = create_data_loader(image_list=test_list)
    evaluate(net, test_dl, device=device)
    
if __name__ == "__main__":
    main()

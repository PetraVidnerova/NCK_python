import logging
from tqdm import tqdm

import torch

def get_class_name(module, classname):
    mod = __import__(module, fromlist=[classname])
    return getattr(mod, classname)

def train(net,
          train_dl,
          val_dl, 
          epochs, 
          optimizer = None,
          optimizer_kwargs = None,
          criterion = None,
          device="cpu"
):

    learning_rate = 1e-4

    net.to(device=device)
    net.train()

    if optimizer is not None:
        optimizer_class = get_class_name("torch.optim", optimizer)
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        optimizer = optimizer_class(net.parameters(),
                                    **optimizer_kwargs)
    else:
        optimizer = torch.optim.Adam()
    logging.info(f"Running with optimizer: {optimizer.__class__.__name__}") 
    
        
    if criterion is None:
        raise ValueError("You have to specify the criterion.")
    criterion = get_class_name("torch.nn", criterion)()
    logging.info(f"Running with criterion: {criterion.__class__.__name__}") 

    logging.info(f"Target number of epochs: {epochs}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    for epoch in range(epochs):
        loss_ep = 0

        print(f"Epoch {epoch} ... ")
        with tqdm(total=len(train_dl)) as t:
            for batch_idx, (data, targets) in enumerate(train_dl):
                data = data.to(device=device)
                targets = targets.to(device=device)
                ## Forward Pass
                optimizer.zero_grad()
                scores = net(data)
                loss = criterion(scores,targets)
                loss.backward()
                optimizer.step()
                loss_ep += loss.item()
                t.update()
            
        print(f"Loss in epoch {epoch} :::: {loss_ep/len(train_dl)}")

        with torch.no_grad():
            # num_correct = 0
            # num_samples = 0
            val_loss = 0
            print("Computing validation accuracy ...")
            with tqdm(total=len(val_dl)) as t:
                for batch_idx, (data,targets) in enumerate(val_dl):
                    data = data.to(device=device)
                    targets = targets.to(device=device)
                    ## Forward Pass
                    scores = net(data)
                    val_loss += criterion(scores, targets).item()
                    t.update()
            print(
                f"VAL loss: {val_loss/len(val_dl)}"
            )
        scheduler.step(val_loss/len(val_dl))

    return net


def evaluate(net, test_dl, criterion, device="cpu"):
    net.eval()
    val_loss = 0
    if criterion is None:
        raise ValueError("You have to specify the criterion.")
    criterion = get_class_name("torch.nn", criterion)()

    with torch.no_grad():
        for data, targets in tqdm(test_dl):
            data = data.to(device=device)
            targets = targets.to(device=device)
            # Forward Pass
            scores = net(data)
            # geting predictions
            val_loss += criterion(scores, targets).item()

        print(
            f"TEST loss: {val_loss/len(test_dl)}"
        )
    return val_loss/len(test_dl)

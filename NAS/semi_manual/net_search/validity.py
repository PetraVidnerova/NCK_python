import torch
from time import time

def check_validity(y_true, y_pred):

    RX, RY, RZ = 0, 1, 2
    X, Y, Z = 3, 4, 5
    
    err = (y_true - y_pred).abs()

    rx_ry = err[:, RX] + err[:, RY]
    cond = torch.logical_and(
      rx_ry < 0.052,
      (err[:, X] + err[:, Y]) < 0.007  
    )
    cond3_ur = err[:, Z] < 0.04
    cond3_ku = err[:, Z] + 0.075*rx_ry < 0.008

    
    return (
      torch.logical_and(cond, cond3_ur),
      torch.logical_and(cond, cond3_ku)
    )
    


def check_validity_orig(err):
  
    err_rx, err_ry, err_rz, err_x, err_y, err_z = err.numpy()

    cond1 = abs(err_rx) + abs(err_ry) < 0.052

    cond2 = abs(err_x) + abs(err_y) < 0.007

    cond3_ur = abs(err_z) < 0.04

    cond3_ku = abs(err_z) + 0.075*(abs(err_rx) + abs(err_ry)) < 0.008

    valid_ur = cond1 and cond2 and cond3_ur

    valid_ku = cond1 and cond2 and cond3_ku

    return valid_ur, valid_ku  



if __name__ == "__main__":

    from data_robot import create_data_loader
    import sys 
    
    test_dl = create_data_loader(1024, train_val=False, test=True)[2]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    net = torch.load(sys.argv[1],  map_location=torch.device(device))
    net = net.to(device)
    net.eval()
    
    start = time()
    num = 0
    ur_correct = 0
    ku_correct = 0 
    for images, labels in test_dl:

        images = images.to(device)
        predictions = net(images).detach().cpu()

        for label, prediction in zip(labels, predictions):

            print(prediction)
            err = label - prediction
            valid_ur, valid_ku = check_validity_orig(err)
            num += 1
            ur_correct += valid_ur
            ku_correct += valid_ku

    print(f"UR: {ur_correct/num:.2f}")
    print(f"KU: {ku_correct/num:.2f}")
    print(f"{time()-start} secs")

    start = time()
    num = 0
    ur_correct = 0
    ku_correct = 0
    for images, labels in test_dl:
        images = images.to(device)
        predictions = net(images).cpu()

        val_ur, val_ku = check_validity(labels, predictions)
        num += len(val_ur)
        ur_correct += val_ur.sum()
        ku_correct += val_ku.sum()
        

    print(f"UR: {ur_correct/num:.2f}")
    print(f"KU: {ku_correct/num:.2f}")
    print(f"{time()-start} secs")

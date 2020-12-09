import torch
from torch import nn
from typing import List, Tuple


def predict(path_model: str, testloader, device=None) -> Tuple[list, list, list]:
    """

    :param path_model:
    :param testloader:
    :param device:
    :return:
    """
    model = torch.load(path_model)
    model.eval()  # to set dropout and batch normalization layers to evaluation mode
    loss_function = nn.MSELoss()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prediction = []
    mse_list = []
    gt = []
    for seq, labels in testloader:
        x, y = seq.to(device, dtype=torch.long), labels.to(device, dtype=torch.long)
        y_pred = model(x.float())
        loss_val = loss_function(y_pred.float(), y.float())
        prediction += [y_pred]
        gt += [y]
        mse_list += [loss_val.data]

    return prediction, gt, mse_list

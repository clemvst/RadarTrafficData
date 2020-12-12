import torch
from torch import nn
from typing import List, Tuple
import matplotlib.pyplot as plt

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
        #x, y = seq.to(device, dtype=torch.long), labels.to(device, dtype=torch.long)
        y_pred = model(seq.float())
        loss_val = loss_function(y_pred.float(), labels.float())
        prediction += [y_pred]
        gt += [labels]
        mse_list += [loss_val.data]

    return prediction, gt, mse_list

def plot_predict(dic_model,valloader,seq_len,lcolor=None):
    if lcolor is None:
        lcolor=["blue","red","green","orange","purple"]
    for seq,label in valloader:
        fig, ax = plt.subplots()
        for i,name_model in enumerate(dic_model): #dic torch_name, torchmodel
            model=dic_model[name_model]
            pred=model.predict(seq,seq_len)
            print(pred.shape)
            print(label.shape)
            xpred=pred.squeeze(0).squeeze(0).detach().numpy()
            ax.plot([i for i in range(seq_len)],xpred,lcolor[i], label="pred_{}".format(name_model))
        xlab = label.squeeze(0).detach().numpy()
        ax.plot([i for i in range(seq_len)],xlab,label="label")
        ax.legend()
        plt.show()
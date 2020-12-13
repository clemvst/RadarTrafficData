import torch
from torch import nn
from typing import List, Tuple
import matplotlib.pyplot as plt

def predict(path_model: str, testloader, device=None, is_interval=False, sample_nbr=7) -> Tuple[list, list, list]:
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
        y_pred = model(seq.float())
        loss_val = loss_function(y_pred.float(), labels.float())
        if is_interval:
            y_pred = [model(seq.float()) for i in range(sample_nbr)]
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
        
        
        
def get_confidence_intervals(preds_test, ci_multiplier):
    """
    Function inspired (BUT WITH MANY CHANGES) by Piero Esposito's functions for Bayesian Model
    
    https://towardsdatascience.com/bayesian-lstm-on-pytorch-with-blitz-a-pytorch-bayesian-deep-learning-library-5e1fec432ad3
    """

    preds_test = torch.cat([preds_test[i].unsqueeze(0) for i in range(len(preds_test))], dim=0)

    pred_mean = preds_test.mean(0)
    pred_std = preds_test.std(0).detach().cpu().numpy()

    pred_std = torch.tensor((pred_std))
    
    upper_bound = pred_mean + (pred_std * ci_multiplier)
    lower_bound = pred_mean - (pred_std * ci_multiplier)
    
    #gather confidence intervals

    pred_mean_final = pred_mean.unsqueeze(1).detach().cpu().numpy()

    upper_bound_unscaled = upper_bound.unsqueeze(1).detach().cpu().numpy()
    
    lower_bound_unscaled = lower_bound.unsqueeze(1).detach().cpu().numpy()
    
    return pred_mean_final, upper_bound_unscaled, lower_bound_unscaled
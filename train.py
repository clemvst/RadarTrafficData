import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from typing import Tuple


def train(model, trainloader, valloader, lr, n_epochs, device=None,ite_print=10) -> Tuple[list, list, list]:
    """
    :param model: a toch model
    :param lr: learning rate, float
    :param n_epochs: number of epochs
    :param device:
    :return:

    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    iteration = []
    loss_train_list = []
    loss_val_list = []
    for i in range(n_epochs):
        for seq, labels in trainloader:
            x, y = seq.to(device, dtype=torch.long), labels.to(device, dtype=torch.long)
            x = Variable(x)
            y = Variable(y)
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(x.float())

            single_loss = loss_function(y_pred.float(), y.float())
            single_loss.backward()
            optimizer.step()

        if i % ite_print == 1:
            # print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
            loss_val_l = []
            for seqval, labelval in valloader:  # maybe change the batch size of val an we avoid using a for loop ?
                xval, yval = seqval.to(device, dtype=torch.long), labelval.to(device, dtype=torch.long)

                xval = Variable(xval)
                yval = Variable(yval)
                output_val = model(xval.float())
                loss_val = loss_function(output_val.float(), yval.float())
                loss_val_l += [loss_val.data]
            iteration += [i]
            loss_train_list += [single_loss.item()]
            loss_val_list += [np.mean(loss_val_l)]
            print("epoch {} loss train {} loss val {}".format(i, single_loss.data, np.mean(loss_val_l)))
        # print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
    return iteration,loss_train_list,loss_val_list
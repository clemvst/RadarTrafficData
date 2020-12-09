import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from typing import Tuple

from constant import MODEL_DIR, CHECKPOINT_DIR, BAD_EPOCH


def load_from_checkpoint(model, optimizer, path_checkpoint):
    """

    :param model:
    :param optimizer:
    :param path_checkpoint:
    :return:
    """
    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, epoch, loss


def train(model, trainloader, valloader, lr: float, n_epochs: int, name_model="model", device=None, ite_print=10,
          save=True) -> Tuple[list, list, list]:
    """
    Trains a chosen model with given training and validation datasets and hyperparameters
    
    :param model: a toch model
    :param trainloader:
    :param valloader:
    :param lr: learning rate, float
    :param n_epochs: number of epochs
    :param device: cuda or cpu
    :param ite_print: path of iterations to print (ex: print every 5 epochs)
    
    :return: tuple with list of epochs iterations, list of train loss values, list of val loss values

    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    iteration = []
    loss_train_list = []
    loss_val_list = []
    best_val_loss = 20.0
    bad_epochs = 0
    for i in range(n_epochs):
        for seq, labels in trainloader:
            #x, y = seq.to(device, dtype=torch.long), labels.to(device, dtype=torch.long)
            #x = Variable(x)
            #y = Variable(y)
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                               torch.zeros(1, 1, model.hidden_layer_size)) #TODO is it useful ??
            y_pred = model(seq.float())

            single_loss = loss_function(y_pred.float(), labels.float())
            single_loss.backward()
            optimizer.step()

        if i % ite_print == 1:
            loss_val_l = []
            for seqval, labelval in valloader:  # maybe change the batch size of val an we avoid using a for loop ?
                #xval, yval = seqval.to(device, dtype=torch.long), labelval.to(device, dtype=torch.long)

                #xval = Variable(xval)
                #yval = Variable(yval)
                output_val = model(seqval.float())
                loss_val = loss_function(output_val.float(), labelval.float())
                loss_val_l += [loss_val.data]
            iteration += [i]
            loss_train_list += [single_loss.item()]
            loss_val_list += [np.mean(loss_val_l)]
            print("epoch {} loss train {} loss val {}".format(i, single_loss.data, np.mean(loss_val_l)))

            if loss_val.data < best_val_loss:
                if save:
                    torch.save(model, MODEL_DIR + name_model + ".pt")
                    torch.save({
                        'epoch': i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': single_loss
                    }, CHECKPOINT_DIR + name_model + "_checkpoint_{}.pt".format(i))
                best_val_loss = loss_val.data
                bad_epochs = 0
            else:
                bad_epochs += 1

            if bad_epochs == BAD_EPOCH:
                break

    return iteration, loss_train_list, loss_val_list

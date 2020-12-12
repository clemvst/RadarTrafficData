import random
import os, errno
import sys
from tqdm import trange
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from constant import MODEL_DIR, CHECKPOINT_DIR


class lstm_encoder(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, input_size, hidden_size, num_layers=1):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # define LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers)

    def forward(self, x_input):
        '''
        : param x_input:               input of shape (seq_len, # in batch, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence
        '''

        lstm_out, self.hidden = self.lstm(x_input) #.view(x_input.shape[0], x_input.shape[1], self.input_size)

        return lstm_out, self.hidden

    def init_hidden(self, batch_size):
        '''
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state
        '''

        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


class lstm_decoder(nn.Module):
    ''' Decodes hidden state output by encoder '''

    def __init__(self, input_size, hidden_size, num_layers=1):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x_input, encoder_hidden_states):
        '''
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence

        '''

        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(0))

        return output, self.hidden


class lstm_seq2seq(nn.Module):
    ''' train LSTM encoder-decoder and make predictions '''

    def __init__(self, input_size, hidden_size):

        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        '''

        super(lstm_seq2seq, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = lstm_encoder(input_size=input_size, hidden_size=hidden_size)
        self.decoder = lstm_decoder(input_size=input_size, hidden_size=hidden_size)

    def train_model(self, trainloader,valloader, n_epochs, target_len, batch_size,
                    training_prediction='recursive', teacher_forcing_ratio=0.5, learning_rate=0.01,
                    save=True,name_model="model_encoder_decoder",
                    dynamic_tf=False,ite_print=1):

        '''
        train lstm encoder-decoder
        :param trainloader : input,label avec label (feat,seq)

        : param input_tensor:              input data with shape (seq_len, # in batch, number features); PyTorch tensor
        : param target_tensor:             target data with shape (seq_len, # in batch, number features); PyTorch tensor
        : param n_epochs:                  number of epochs
        : param target_len:                number of values to predict
        : param batch_size:                number of samples per gradient update
        : param training_prediction:       type of prediction to make during training ('recursive', 'teacher_forcing', or
        :                                  'mixed_teacher_forcing'); default is 'recursive'
        : param teacher_forcing_ratio:     float [0, 1) indicating how much teacher forcing to use when
        :                                  training_prediction = 'teacher_forcing.' For each batch in training, we generate a random
        :                                  number. If the random number is less than teacher_forcing_ratio, we use teacher forcing.
        :                                  Otherwise, we predict recursively. If teacher_forcing_ratio = 1, we train only using
        :                                  teacher forcing.
        : param learning_rate:             float >= 0; learning rate
        : param dynamic_tf:                use dynamic teacher forcing (True/False); dynamic teacher forcing
        :                                  reduces the amount of teacher forcing for each epoch
        : return losses:                   array of loss function for each epoch
        '''

        # initialize array of losses
        losses = np.full(n_epochs, np.nan)

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        loss_val_list=[]
        iteration=[]

        # calculate number of batch iterations
      #  n_batches = int(input_tensor.shape[1] / batch_size)

        with trange(n_epochs) as tr:
            for i,it in enumerate(tr):
                batch_loss = 0.
                batch_loss_tf = 0.
                batch_loss_no_tf = 0.
                num_tf = 0
                num_no_tf = 0
                best_val_loss = 20.0
                for input_batch,target_batch in trainloader:
                    # select data
                    #input_batch = input_tensor[:, b: b + batch_size, :]
                    #target_batch = target_tensor[:, b: b + batch_size, :]
                    input_batch=input_batch.unsqueeze(1) #feat,batch,seq
                    target_batch=target_batch.unsqueeze(1) #feat,batch,seq
                    input_batch=input_batch.view(input_batch.shape[-1],input_batch.shape[1],input_batch.shape[0]) #seq,batch,feat
                    target_batch = target_batch.view(target_batch.shape[-1], target_batch.shape[1],target_batch.shape[0])
                    #input_batch.unsqueeze(1)
                    #target_batch.unsqueeze(1) #(feat,batch,seq)
                    #print(input_batch.shape)
                    #print(target_batch.shape)
                    # outputs tensor
                    outputs = torch.zeros(target_len, batch_size, input_batch.shape[2]) #seq,batch,feat

                    # initialize hidden state
                    encoder_hidden = self.encoder.init_hidden(batch_size)
                    #print(target_len)
                    # zero the gradient
                    optimizer.zero_grad()

                    # encoder outputs
                    encoder_output, encoder_hidden = self.encoder(input_batch)

                    # decoder with teacher forcing
                    decoder_input = input_batch[-1, :, :]  # shape: (batch_size, input_size)
                    decoder_hidden = encoder_hidden

                    if training_prediction == 'recursive':
                        # predict recursively
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[t] = decoder_output
                            decoder_input = decoder_output

                    if training_prediction == 'teacher_forcing':
                        # use teacher forcing
                        if random.random() < teacher_forcing_ratio:
                            for t in range(target_len):
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[t] = decoder_output
                                decoder_input = target_batch[t, :, :]

                        # predict recursively
                        else:
                            for t in range(target_len):
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[t] = decoder_output
                                decoder_input = decoder_output

                    if training_prediction == 'mixed_teacher_forcing':
                        # predict using mixed teacher forcing
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[t] = decoder_output

                            # predict with teacher forcing
                            if random.random() < teacher_forcing_ratio:
                                decoder_input = target_batch[t, :, :]

                            # predict recursively
                            else:
                                decoder_input = decoder_output

                    # compute the loss
                    loss = criterion(outputs, target_batch)
                    batch_loss += loss.item()

                    # backpropagation
                    loss.backward()
                    optimizer.step()

                if i % ite_print == 1:
                    loss_val_l =0
                    for seqval, labelval in valloader:  # maybe change the batch size of val an we avoid using a for loop ?
                        #seq_batch = seqval.unsqueeze(1)  # feat,seq
                        #seq_batch = target_batch.unsqueeze(1)  # feat,seq
                        #seqval = seqval.view(input_batch.shape[-1],)  # seq,feat
                        ##                                target_batch.shape[0])
                        labelval = labelval.unsqueeze(1)  # feat,batch,seq
                        labelval = labelval.view(labelval.shape[-1], labelval.shape[1],
                                                         labelval.shape[0])
                        output_val = self.predict(seqval,target_len=target_len)
                        loss_val = criterion(output_val.float(), labelval.float())
                        loss_val_l += loss_val.data
                    loss_val_list += [loss_val_l/len(valloader)]
                    batch_loss /= len(trainloader)
                    losses[it] = batch_loss
                    iteration += [i]
                    #print("epoch {} loss train {} loss val {}".format(i, single_loss.data, np.mean(loss_val_l)))
                # loss for epoch
                    print("epoch {} loss train {} loss val {}".format(i, batch_loss, loss_val_l))
                # dynamic teacher forcing
                    if loss_val.data < best_val_loss:
                        if save:
                            torch.save(self, MODEL_DIR + name_model + ".pt")
                            torch.save({
                                'epoch': i,
                                'model_state_dict': self.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': batch_loss
                            }, CHECKPOINT_DIR + name_model + "_checkpoint_{}.pt".format(i))
                        best_val_loss = loss_val.data

                if dynamic_tf and teacher_forcing_ratio > 0:
                    teacher_forcing_ratio = teacher_forcing_ratio - 0.02

                    # progress bar
                tr.set_postfix(loss="{0:.3f}".format(batch_loss))

        return iteration,losses,loss_val_list

    def predict(self,input_batch,target_len,batch_size=1):
        """
        :param input_batch: dim (seq,feat)
        :param target_len:
        :return:
        """
        input_batch = input_batch.unsqueeze(1)  # feat,batch,seq
        input_batch = input_batch.view(input_batch.shape[-1], input_batch.shape[1],
                                       input_batch.shape[0])  # seq,batch,feat

        outputs = torch.zeros(target_len, batch_size, input_batch.shape[2])  # seq,batch,feat

        # initialize tensor for predictions
        encoder_output, encoder_hidden = self.encoder(input_batch)
        #outputs = torch.zeros(target_len, input_batch.shape[2])
        # decode input_tensor
        decoder_input = input_batch[-1, :, :]
        decoder_hidden = encoder_hidden
        for t in range(target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output.squeeze(0)
            decoder_input = decoder_output

        #np_outputs = outputs.detach().numpy()

        return outputs

    def predict_old(self, input_tensor, target_len):

        '''
        : param input_tensor:      input data (seq_len, input_size); PyTorch tensor
        : param target_len:        number of target values to predict
        : return np_outputs:       np.array containing predicted values; prediction done recursively
        '''


        # encode input_tensor
        input_tensor = input_tensor.unsqueeze(1)  # add in batch size of 1
        encoder_output, encoder_hidden = self.encoder(input_tensor)

        # initialize tensor for predictions
        outputs = torch.zeros(target_len, input_tensor.shape[2])

        # decode input_tensor
        decoder_input = input_tensor[-1, :, :]
        decoder_hidden = encoder_hidden

        for t in range(target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output.squeeze(0)
            decoder_input = decoder_output

        np_outputs = outputs.detach().numpy()

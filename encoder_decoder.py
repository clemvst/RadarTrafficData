from random import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from tqdm import trange

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

        lstm_out, self.hidden = self.lstm(x_input.view(x_input.shape[1], x_input.shape[0],-1)) #TODO check
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
        self.linear = nn.Linear(hidden_size, input_size) #TODO I do not understand this dimensions

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


class lstm_wrapper(nn.Module):
    ''' train LSTM encoder-decoder and make predictions '''

    def __init__(self, input_size, hidden_size,target_len=96,teacher_forcing_ratio=0.3):
        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        '''

        super(lstm_wrapper, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = lstm_encoder(input_size=input_size, hidden_size=hidden_size)
        self.decoder = lstm_decoder(input_size=input_size, hidden_size=hidden_size)
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.target_len=target_len
        self.batch_size=1
        self.optimizer = optim.Adam(self.parameters())

    def forward(self,input_batch,target_batch):
        """

        :param input_batch: Tensor, input in the encoder, contains times series + features
        :param decoder_features: Contains features (if there are any)
        :return:
        """
        self.optimizer.zero_grad()
        # initialize hidden state

        # encoder outputs
        encoder_output, encoder_hidden = self.encoder(input_batch)
        # outputs tensor
        outputs = torch.zeros(self.target_len, self.batch_size, input_batch.shape[2])
        # decoder with teacher forcing
        decoder_input = input_batch[-1, :, :]  # shape: (batch_size, input_size)  #TODO understand this line I have seen
        #A LOT OF THESE INTIALIZATION JUST BE SUure
        # initialization is that and not encoder_output..
        decoder_hidden = encoder_hidden

        # use teacher forcing
        if random() < self.teacher_forcing_ratio:
            for t in range(self.target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[t] = decoder_output
                decoder_input = target_batch[t, :, :]

        # predict recursively
        else:
            for t in range(self.target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[t] = decoder_output
                decoder_input = decoder_output



    def train_model(self, trainloader, n_epochs, target_len, batch_size, teacher_forcing_ratio=0.5,
                    learning_rate=0.01,save=False,name_model="model",nfeatures=1):

        '''
        This model  is taken from https://github.com/lkulowski/LSTM_encoder_decoder/blob/master/code/lstm_encoder_decoder.py
        train lstm encoder-decoder

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
        #TODO add features to be taken into account in the decoder ...

        # initialize array of losses
        losses = np.full(n_epochs, np.nan)
        loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # calculate number of batch iterations
        #n_batches = int(input_tensor.shape[1] / batch_size)

        with trange(n_epochs) as tr:
            for it in tr:

                batch_loss = 0.
                batch_size=0
                for input_batch,target_batch in trainloader:
                    #input_batch=input_batch.unsqueeze(0) #dimension should be (batch_size,feature,seq_len)
                    #target_batch = input_batch.unsqueeze(0)
                    print("input",input_batch.shape)
                    print("target",target_batch.shape)
                    batch_size+=1
                    # select data
                    #input_batch = input_tensor[:, b: b + batch_size, :]
                    #target_batch = target_tensor[:, b: b + batch_size, :]

                    # outputs tensor
                    outputs = torch.zeros(batch_size,nfeatures,target_len) #TODO MODIFY IF FEATURES

                    # initialize hidden state
                    encoder_hidden = self.encoder.init_hidden(batch_size) #TODO understand this line I have seen

                    # zero the gradient
                    self.optimizer.zero_grad()

                    # encoder outputs
                    encoder_output, encoder_hidden = self.encoder(input_batch)

                    # decoder with teacher forcing
                    decoder_input = input_batch[:, :,-1]  # shape: (batch_size, nfeatures,seq_len)
                    decoder_hidden = encoder_hidden

                    # use teacher forcing
                    if random() < teacher_forcing_ratio:
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            print(outputs.shape)
                            print(decoder_output.shape)
                            outputs[:,:,t] = decoder_output
                            decoder_input = target_batch[:,:,t]

                    # predict recursively
                    else:
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            print(outputs.shape)
                            print(decoder_output.shape)
                            outputs[:,:,t] = decoder_output
                            decoder_input = decoder_output[:,:,t]


                    # compute the loss
                    print("output dim",outputs.shape,target_batch.shape)
                    loss =  loss_function(outputs, target_batch)
                    batch_loss += loss.item()

                    # backpropagation
                    loss.backward()
                    self.optimizer.step()

                # loss for epoch
                batch_loss /= batch_size
                losses[it] = batch_loss



                    # progress bar
                tr.set_postfix(loss="{0:.3f}".format(batch_loss))

        return losses

    def predict(self, input_tensor, target_len):

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

        return np_outputs
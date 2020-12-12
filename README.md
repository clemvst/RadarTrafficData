# RadarTrafficData
Final ML project for the Machine Learning course of Ecole des Mines de Nancy

Kaggle link: https://www.kaggle.com/vinayshanbhag/radar-traffic-data

## Prepare the environment
Install Anaconda 
 `conda env create -f environment.yaml` to create the env
 `conda activate dl_env` to activate the env 
 
 `python -m ipykernel install --user --name=dl_env` to add the configurated env as a jupyter kernel
 
 
##TODO : 

- Check with a more complex task if the training still works
-Add features use in the encoder/decoder model !  refer to encoder_decoder_clean.py
- Look for the more complex model encoder-decoder. The model implemented is a seq2seq. Thus the input
sequence and the sequence to predict have the same dimension. There is some solutions to be able to predict
on a sequence which does not have the same size as the input data, I have seen solution in Keras. With Repeat Vector
(https://machinelearningmastery.com/lstm-autoencoders/)
Feature to add : day of the week, then we can also use the categorical variable for the radar_name. 
And create a dataset on multiple radar then check that 

Check if it is easey to implement otherwise maybe let's focus on the features adding.
-Features : 
https://github.com/gautham20/pytorch-ts/tree/master/ts_models
https://towardsdatascience.com/encoder-decoder-model-for-multistep-time-series-forecasting-using-pytorch-5d54c6af6e60
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

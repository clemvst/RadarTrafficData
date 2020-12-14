import argparse

import torch
from sklearn.model_selection import train_test_split

from dataset import RadarCollate, RadarDataset
from open_data import open_data, create_global_batch, get_df_stats, apply_norm
from torch.utils.data import DataLoader

from simple_model import LSTM
from train import train


def _argparser():
    parser = argparse.ArgumentParser(description="Argument GAN train")
    parser.add_argument('--dataset', type=str, default="/Users/iris/Documents/radar_deep/Radar_Traffic_Counts.csv",
                        help="path to the dataset ")
    parser.add_argument('--radar', type=str, default=' CAPITAL OF TEXAS HWY / LAKEWOOD DR',
                        help="radar name, could be None ")
    parser.add_argument('--day_input', type=int, default=7,
                        help="nber of day to be taken into the input")
    parser.add_argument('--day_label', type=int, default=1,
                        help="nber of day to be predict")
    parser.add_argument('--year', type=int, default=2018,
                        help="the data will be taken from year")
    parser.add_argument('--max_total_day', type=int, default=365,
                        help="total nber from the first day of the year that are considered for building the dataset")
    parser.add_argument('--name_model', type=str, default="simple_model_lstm",
                        help="total nber from the first day of the year that are considered for building the dataset")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning_rate")
    parser.add_argument('--epochs', type=int, default=300,
                        help="nber of epochs")

    return parser.parse_args()

def main(path_dataset,radar_name,days_input,window_label,year,total_len_days=365,batch_size=1,name_model="",
         learning_rate=0.01,epochs=100):
    df1=open_data(path_dataset, direction="NB", radar=radar_name, year=year)
    batch_df = create_global_batch(df1, window_x_day=days_input, window_label_day=window_label, gap_acquisition=1,
                                   tot_len_day=total_len_days)
    df_mean, df_std = get_df_stats(batch_df)
    new_data = apply_norm(batch_df, df_mean, df_std)
    fullset = RadarDataset(dataframe=batch_df, transform=None)
    ptrain, pval, ptest = 0.7, 0.15, 0.15
    trainglobdataset, testdataset = train_test_split(fullset, test_size=ptest, shuffle=False)
    traindataset, valdataset = train_test_split(trainglobdataset, test_size=pval / (1 - ptest), shuffle=False)
    collate_fn = RadarCollate()
    trainloader = DataLoader(traindataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    valloader = DataLoader(valdataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    input_dim = days_input*24*4
    hidden_dim = 256
    output_dim = window_label*24*4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTM(input_size=input_dim, hidden_size=hidden_dim, output_size=output_dim)
    iteration_sm, loss_train_list_sm, loss_val_list_sm = train(model, trainloader, valloader, lr=learning_rate, n_epochs=epochs,
                                                               name_model=name_model, device=device, ite_print=5,
                                                               save=True)
    return iteration_sm,loss_train_list_sm,loss_val_list_sm

if __name__ == '__main__':
    parser = _argparser()
    main(parser.dataset,parser.radar,parser.day_input,parser.day_label,parser.year,total_len_days=parser.max_total_day,name_model=parser.name_model,
         learning_rate=parser.lr,epochs=parser.epochs)
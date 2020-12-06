import datetime
from datetime import timedelta
import pandas as pd
from typing import Tuple


def open_data(str_csv, direction: str, radar: str, year=None):
    """
    Open the csv, and filter the data given the values of the argument
    :param str_csv: path to csv
    :param year: year to keep
    :param radar: radar name to keep
    :param direction: direction of the radar to keep
    :return: a pandas DataFrame
    """
    # TODO keep the other parameters as week days ect..
    df = pd.read_csv(str_csv, parse_dates={"global_date": [3, 4, 5, 9]}, keep_date_col=True)  # open the csv
    df = df[(df['location_name'] == radar)]
    if year is not None:
        df = df.loc[df["Year"] == str(year)]
    df = df.loc[df["Direction"] == direction]
    df = df.groupby("global_date").agg({"Volume": "sum", "Day of Week": lambda x: list(set(x))[0]}).sort_values(
        "global_date").reset_index()
    df["date"] = df.apply(lambda x: x.global_date.date(), axis=1)  # we use datetime functions
    return df


def create_subd_df(df: pd.DataFrame, begin_day: datetime.date, window_x_day: int, window_label_day: int) :
    """
    Given a begin date, extract the sub_df of the data for x and y.
    :return:
    the sub DataFrame for x, the sub DataFrame for label
    """
    end_x_window = begin_day + timedelta(days=window_x_day)
    end_label_window = end_x_window + timedelta(days=window_label_day)
    df_x = df.loc[(df["date"] < end_x_window) & (df["date"] >= begin_day)]
    df_label = df.loc[(df["date"] < end_label_window) & (df["date"] >= end_x_window)]
    # we need to check that we have data for all the dates :
    if len(df_x["date"].unique()) != window_x_day:
        print("We do not have all the dates for the time period in x , {} {}".format(window_x_day, df_x["date"].unique()))
        return None, None
    elif len(df_label["date"].unique()) != window_label_day:
        print("We do not have all the dates for the time period in label , {} {}".format(window_label_day,
                                                                                     df_label["date"].unique()))
        return None, None
    else:
        # print(df_label)
        return df_x, df_label


def create_global_batch(df, window_x_day: int, window_label_day: int, gap_acquisition: int, tot_len_day=365,
                        features=None): #TODO Features not working yet
    """
    We always think it terms of days
    :param df:
    :param window_x_day:
    :param window_label_day:
    :param gap_acquisition:
    :param tot_len_day:
    :param features: To implement
    :return:
    """
    columns_name = ["vol_data_x", "vol_label_y"]
    df = df.sort_values("global_date").reset_index()  # we ensure that the data frame is created by time
    first_day = df["date"].iloc[0] + timedelta(
        days=1)  # Sometimes in the first day the acquisition does start at midnight. But will be ok for the second day
    end_period = first_day + timedelta(days=tot_len_day)
    if df["date"].iloc[-1] < end_period:
        end_period = df["date"].iloc[-1]
    batch_df = pd.DataFrame(columns=columns_name)
    i = 0
    while first_day < end_period - timedelta(days=window_label_day + window_x_day):
        print("Building batch {} \n x begin {} label begin {} end period {} ".format(i, first_day,
                                                                                     first_day + timedelta(
                                                                                         days=window_x_day),
                                                                                     end_period - timedelta(
                                                                                         days=window_label_day + window_x_day)))
        df_x, df_y = create_subd_df(df, begin_day=first_day, window_x_day=window_x_day,
                                    window_label_day=window_label_day)
        first_day = first_day + timedelta(days=gap_acquisition)
        if df_x is not None: # ensure that we have data for all the dates
            dic_data = {"vol_data_x": df_x["Volume"].to_numpy(), "vol_label_y": df_y["Volume"].to_numpy()}
            if features is not None:
                # features should fit with the df_x column name
                assert features in df_x.columns, "Cannot add the features {} as it is not in the sub_df_x cols " \
                                                 "{}".format(features, df_x.columns)
                dic_data.update({features: df_x[features]})
            batch_df = batch_df.append(dic_data, ignore_index=True)
    i += 1
    return batch_df

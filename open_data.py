import pandas as pd


def open_data(str_csv, direction, radar, year=None):
    """
    Open the csv, and filter the data given the values of the argument
    :param str_csv: path to csv
    :param year: year to keep
    :param radar: radar name to keep
    :param direction: direction of the radar to keep
    :return: a pandas DataFrame
    """
    df = pd.read_csv(str_csv, parse_dates={"global_date": [3, 4, 5, 9]}, keep_date_col=True)  # open the csv
    df = df[(df['location_name'] == radar)]
    if year is not None:
        df = df[df["Year"] == 2018]
    df = df[df["Direction"] == direction]
    if (radar is not None) | (year is not None):
        # aggregate the values if we have data
        df = df.groupby("global_date").agg({"Volume": "sum"}).sort_values("global_date").reset_index()
    return df


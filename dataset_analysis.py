import argparse
import pandas as pd
from data_utils import Preprocessor
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '-d', type=str,
                        help='MetroTraffic, BeijingAirQuality, AustraliaTourism, WebTraffic, StoreItems', required=True)
    parser.add_argument('-propCycEnc', type=bool, default=False)
    args = parser.parse_args()
    dataset = args.dataset
    preprocessor = Preprocessor(dataset, args.propCycEnc)
    df = preprocessor.df_cleaned
    df_orig = preprocessor.df_orig
    plt.plot(df['weather_description_sine'])
    plt.show()
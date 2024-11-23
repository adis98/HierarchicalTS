import pandas as pd
from matplotlib import pyplot as plt
from data_utils import Preprocessor
import numpy as np
import os

if __name__ == "__main__":
    xcorrdtable = pd.DataFrame(
        columns=['Dataset', 'Method', 'Level', 'Avg. xcorrD', 'Std. xcorrD'])
    for dataset in ["AustraliaTourism", "MetroTraffic", "RossmanSales", "BeijingAirQuality", "PanamaEnergy"]:
        preprocessor = Preprocessor(dataset, False)
        for method in ["timegan", "timeweaver", "tsdiff-0.0", "tsdiff-0.5", 'tsdiff-1.0', 'tsdiff-2.0', 'algo-1',
                       'algo-8', 'algo-16', 'algo-32']:
            for mask in ['C', 'M', 'F']:
                real = pd.read_csv(f'generated/{dataset}/{mask}/real.csv')
                non_hier_cols = [col for col in real.columns if
                                 col not in preprocessor.hierarchical_features_uncyclic and col != 'Unnamed: 0']

                filt = real[non_hier_cols]
                xcorr_real = filt.corr()
                MAES = []
                for trial in range(5):
                    if method == "timegan":
                        data = pd.read_csv(f'generated/{dataset}/{mask}/synth_timegan_trial_{trial}_cycStd.csv')
                    elif method == "timeweaver":
                        data = pd.read_csv(f'generated/{dataset}/{mask}/synth_timeweaver_trial_{trial}_cycStd.csv')
                    elif "tsdiff" in method:
                        strength = method.split('-')[1]
                        data = pd.read_csv(
                            f'generated/{dataset}/{mask}/synth_tsdiff_strength_{strength}_trial_{trial}.csv')
                    else:
                        stride = method.split('-')[1]
                        data = pd.read_csv(
                            f'generated/{dataset}/{mask}/synth_hyacinth_pipeline_stride_{stride}_trial_{trial}_cycStd.csv')
                    data = data[non_hier_cols]
                    xcorr_data = data.corr()
                    diff = (xcorr_real - xcorr_data).abs().mean().mean()
                    MAES.append(diff)
                arr = np.array(MAES)
                avg = np.mean(arr)
                std = np.std(arr)
                if "algo" in method:
                    tech = 'hyacinth-' + method.split('-')[1]
                else:
                    tech = method
                row = {"Dataset": dataset, "Method": tech, "Level": mask, 'Avg. xcorrD': avg, 'Std. xcorrD': std}
                xcorrdtable.loc[len(xcorrdtable)] = row
    path = "experiments/xcorrdtable/"
    if not os.path.exists(path):
        os.makedirs(path)
    final_path = os.path.join(path, "xcorrdtable.csv")
    xcorrdtable.to_csv(final_path)


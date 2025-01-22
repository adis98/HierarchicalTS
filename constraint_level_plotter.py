import pandas as pd
from data_utils import Preprocessor
from metasynth import metadataMask
from matplotlib import pyplot as plt

if __name__ == "__main__":
    preprocessor = Preprocessor("AustraliaTourism", False)
    df = preprocessor.df_cleaned
    test_df = df.loc[preprocessor.train_indices[-32:] + preprocessor.test_indices]
    test_df_with_hierarchy = preprocessor.cyclicDecode(test_df)
    metadata = test_df_with_hierarchy[preprocessor.hierarchical_features_uncyclic]

    # Generate the binary mask for rows to synthesize
    rows_to_synth = metadataMask(metadata, "F", "AustraliaTourism")

    # Extract the 'Trips' data
    trips_data = test_df_with_hierarchy['Trips']

    # Plot the full signal in gray
    plt.plot(trips_data, color="black", label="Missing Data")

    # Overlay only the available data points in blue (without continuity)
    plt.plot(trips_data[~rows_to_synth], color="orange", label="Available Data")
    plt.xticks([])
    plt.xlabel("Time", fontsize=20)
    # plt.xlabel("Ordered by: Year\u2192Month\u2192State\u2192Region\u2192Purpose ", fontsize=17)
    plt.ylabel("Trips", fontsize=20)
    plt.title("Constraint: (2016, *, *, *, Holiday)", fontsize=20)
    plt.legend(fontsize=20)
    # plt.show()
    plt.savefig("ftype.pdf", bbox_inches="tight")

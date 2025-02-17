import pandas as pd

if __name__ == "__main__":
    """ENCODING ABLATION"""
    df = pd.read_csv('experiments/ablations/encoding/ablation_encoding.csv')
    for dataset in ['MetroTraffic', 'PanamaEnergy', 'RossmanSales']:
        for encoding in ['OHE', 'STD']:
            for masking in ['C', 'M', 'F']:
                row = df.loc[(df['Dataset'] == dataset) & (df['Mask'] == masking) & (df['Encoding'] == encoding) & (df['stride'] == 8)]
                print(f'& ${row['Avg. MSE'].values[0]: .3f}-{row['Std. MSE'].values[0]: .3f}$', end=' ')
            print(f"& {row["indim"].values[0]}", end=" ")
            print('\\\\')

    """PARALLELISM ABLATION"""
    # df = pd.read_csv('experiments/ablations/parallelism/ablation_parallelism.csv')
    # mapper = {"AR-16": "AR-16", "AR-32": "AR-32", "DNQ": "DNQ", "Pipe": "Pipe", "Pipe-1": "Pipe-1", "Pipe-8": "Pipe-8", "Pipe-16": "Pipe-16", "Pipe-32": "Pipe-32", "AR-8": "AR-8"}
    # for parallel in ["AR-8", "AR-16", "AR-32", "DNQ", "Pipe", "Pipe-1", "Pipe-8", "Pipe-16", "Pipe-32"]:
    #     for dataset in ["PanamaEnergy"]:
    #         for level in ["C", "M", "F"]:
    #             row = df.loc[(df["Parallelism"] == parallel) & (df['Dataset'] == dataset) & (df['Level'] == level)]
    #             val = row["Avg. MSE"].values[0]
    #             std = row["Std. MSE"].values[0]
    #             queries = row["Queries"].values[0]
    #             time = row["Avg. Time"].values[0]
    #             std_time = row["Std. Time"].values[0]
    #
    #             if level == "C":
    #                 print(
    #                     f" {mapper[parallel]} & ${val:.3f} \\pm \\text{{\\scriptsize{{{std:.3f}}}}}$ & ${int(queries)}$ & ${time:.3f} \\pm \\text{{\\scriptsize{{{std_time:.3f}}}}}$",
    #                     end="")
    #             elif level == "M":
    #                 print(
    #                     f" & ${val:.3f} \\pm \\text{{\\scriptsize{{{std:.3f}}}}}$ & ${int(queries)}$ & ${time:.3f} \\pm \\text{{\\scriptsize{{{std_time:.3f}}}}}$",
    #                     end="")
    #             else:
    #                 print(
    #                     f" & ${val:.3f} \\pm \\text{{\\scriptsize{{{std:.3f}}}}}$ & ${int(queries)}$ & ${time:.3f} \\pm \\text{{\\scriptsize{{{std_time:.3f}}}}}$\\\\")

    """bigtable"""
    # df = pd.read_csv('experiments/bigtable/bigtable.csv')
    # # mapper = {"AR-16": "AR-16", "AR-32": "AR-32", "DNQ": "DNQ", "Pipe": "Pipe", "Pipe-1": "Pipe-1", "Pipe-8": "Pipe-8", "Pipe-16": "Pipe-16", "Pipe-32": "Pipe-32", "AR-8": "AR-8"}
    # for method in ["TimeGAN", "TimeWeaver", "TSDiff-0", "TSDiff-0.5", "TSDiff-1.0", "TSDiff-2.0", "Pipe-1", "Pipe-8", "Pipe-16", "Pipe-32"]:
    #     for dataset in ["AustraliaTourism", "MetroTraffic", "BeijingAirQuality", "RossmanSales", "PanamaEnergy"]:
    #         for level in ["C", "M", "F"]:
    #             row = df.loc[(df["Method"] == method) & (df['Dataset'] == dataset) & (df['Level'] == level)]
    #             val = row["Avg. MSE"].values[0]
    #             std = row["Std. MSE"].values[0]
    #             std_decimal = f"{std:.3f}".split(".")[1]
    #             if level == "C" and dataset == "AustraliaTourism":
    #                 print(
    #                     f" {method} & ${val:.3f}_{{.{{{std_decimal}}}}}$",
    #                     end="")
    #             else:
    #                 print(
    #                     f" & ${val:.3f}_{{.{{{std_decimal}}}}}$",
    #                     end=""
    #                 )
    #     print("\\\\")

    # """ACD table"""
    # df = pd.read_csv('experiments/acdtable/acdtable.csv')
    # # mapper = {"AR-16": "AR-16", "AR-32": "AR-32", "DNQ": "DNQ", "Pipe": "Pipe", "Pipe-1": "Pipe-1", "Pipe-8": "Pipe-8", "Pipe-16": "Pipe-16", "Pipe-32": "Pipe-32", "AR-8": "AR-8"}
    # for method in ["TimeGAN", "TimeWeaver", "TSDiff-0.0", "TSDiff-0.5", "TSDiff-1.0", "TSDiff-2.0", "hyacinth-1", "hyacinth-8", "hyacinth-16", "hyacinth-32"]:
    #     for dataset in ["AustraliaTourism", "MetroTraffic", "BeijingAirQuality", "RossmanSales", "PanamaEnergy"]:
    #         for level in ["C", "M", "F"]:
    #             row = df.loc[(df["Method"] == method.lower()) & (df['Dataset'] == dataset) & (df['Level'] == level)]
    #             val = row["Avg. ACD"].values[0]
    #             std = row["Std. ACD"].values[0]
    #             std_decimal = f"{std:.3f}".split(".")[1]
    #             if level == "C" and dataset == "AustraliaTourism":
    #                 print(
    #                     f" {method} & ${val:.3f}_{{.{{{std_decimal}}}}}$",
    #                     end="")
    #             else:
    #                 print(
    #                     f" & ${val:.3f}_{{.{{{std_decimal}}}}}$",
    #                     end=""
    #                 )
    #     print("\\\\")

    # """xcorrD table"""
    # df = pd.read_csv('experiments/xcorrdtable/xcorrdtable.csv')
    # # mapper = {"AR-16": "AR-16", "AR-32": "AR-32", "DNQ": "DNQ", "Pipe": "Pipe", "Pipe-1": "Pipe-1", "Pipe-8": "Pipe-8", "Pipe-16": "Pipe-16", "Pipe-32": "Pipe-32", "AR-8": "AR-8"}
    # for method in ["TimeGAN", "TimeWeaver", "TSDiff-0.0", "TSDiff-0.5", "TSDiff-1.0", "TSDiff-2.0", "hyacinth-1",
    #                "hyacinth-8", "hyacinth-16", "hyacinth-32"]:
    #     for dataset in ["MetroTraffic", "BeijingAirQuality", "RossmanSales", "PanamaEnergy"]:
    #         for level in ["C", "M", "F"]:
    #             row = df.loc[(df["Method"] == method.lower()) & (df['Dataset'] == dataset) & (df['Level'] == level)]
    #             val = row["Avg. xcorrD"].values[0]
    #             std = row["Std. xcorrD"].values[0]
    #             std_decimal = f"{std:.3f}".split(".")[1]
    #             if level == "C" and dataset == "MetroTraffic":
    #                 print(
    #                     f" {method} & ${val:.3f}_{{.{{{std_decimal}}}}}$",
    #                     end="")
    #             else:
    #                 print(
    #                     f" & ${val:.3f}_{{.{{{std_decimal}}}}}$",
    #                     end=""
    #                 )
    #     print("\\\\")

    # "BEST"
    # print("ACD Scores")
    # df = pd.read_csv('experiments/acdtable/acdtable.csv')
    # # mapper = {"AR-16": "AR-16", "AR-32": "AR-32", "DNQ": "DNQ", "Pipe": "Pipe", "Pipe-1": "Pipe-1", "Pipe-8": "Pipe-8", "Pipe-16": "Pipe-16", "Pipe-32": "Pipe-32", "AR-8": "AR-8"}
    # for method in ["TSDiff-0.0", "TSDiff-0.5", "TSDiff-1.0", "TSDiff-2.0", "hyacinth-1", "hyacinth-8", "hyacinth-16", "hyacinth-32"]:
    #     score = 0.0
    #     for dataset in ["AustraliaTourism", "MetroTraffic", "BeijingAirQuality", "RossmanSales", "PanamaEnergy"]:
    #         for level in ["C", "M", "F"]:
    #             row = df.loc[(df["Method"] == method.lower()) & (df['Dataset'] == dataset) & (df['Level'] == level)]
    #             val = row["Avg. ACD"].values[0]
    #             std = row["Std. ACD"].values[0]
    #             std_decimal = f"{std:.3f}".split(".")[1]
    #             score += val
    #     print(f'{method}: {score}: .2f')
    #
    # print("XCORR SCORES")
    # df = pd.read_csv('experiments/xcorrdtable/xcorrdtable.csv')
    # # mapper = {"AR-16": "AR-16", "AR-32": "AR-32", "DNQ": "DNQ", "Pipe": "Pipe", "Pipe-1": "Pipe-1", "Pipe-8": "Pipe-8", "Pipe-16": "Pipe-16", "Pipe-32": "Pipe-32", "AR-8": "AR-8"}
    # for method in ["TSDiff-0.0", "TSDiff-0.5", "TSDiff-1.0", "TSDiff-2.0", "hyacinth-1",
    #                "hyacinth-8", "hyacinth-16", "hyacinth-32"]:
    #     score = 0.0
    #     for dataset in ["AustraliaTourism", "MetroTraffic", "BeijingAirQuality", "RossmanSales", "PanamaEnergy"]:
    #         for level in ["C", "M", "F"]:
    #             row = df.loc[(df["Method"] == method.lower()) & (df['Dataset'] == dataset) & (df['Level'] == level)]
    #             val = row["Avg. xcorrD"].values[0]
    #             std = row["Std. xcorrD"].values[0]
    #             std_decimal = f"{std:.3f}".split(".")[1]
    #             score += val
    #     print(f'{method}: {score}')
    #
    # print("MSESCORES")
    # df = pd.read_csv('experiments/bigtable/bigtable.csv')
    # # mapper = {"AR-16": "AR-16", "AR-32": "AR-32", "DNQ": "DNQ", "Pipe": "Pipe", "Pipe-1": "Pipe-1", "Pipe-8": "Pipe-8", "Pipe-16": "Pipe-16", "Pipe-32": "Pipe-32", "AR-8": "AR-8"}
    # for method in ["TSDiff-0", "TSDiff-0.5", "TSDiff-1.0", "TSDiff-2.0", "Pipe-1", "Pipe-8", "Pipe-16", "Pipe-32"]:
    #     score = 0.0
    #     for dataset in ["AustraliaTourism", "MetroTraffic", "BeijingAirQuality", "RossmanSales", "PanamaEnergy"]:
    #         for level in ["C", "M", "F"]:
    #             row = df.loc[(df["Method"] == method) & (df['Dataset'] == dataset) & (df['Level'] == level)]
    #             val = row["Avg. MSE"].values[0]
    #             std = row["Std. MSE"].values[0]
    #             std_decimal = f"{std:.3f}".split(".")[1]
    #             score += val
    #
    #     print(f'{method}: {score}')

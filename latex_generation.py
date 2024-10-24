import pandas as pd

if __name__ == "__main__":
    """ENCODING ABLATION"""
    # df = pd.read_csv('experiments/ablations/encoding/ablation_encoding.csv')
    # for encoding in ['OHE', 'ORD', 'STD', 'PROP']:
    #     for metric in ['mse', 'acc']:
    #         for dataset in ['AustraliaTourism', 'MetroTraffic']:
    #             for masking in ['C', 'M', 'F']:
    #                 row = df.loc[(df['Dataset'] == dataset) & (df['Mask'] == masking) & (df['Encoding'] == encoding)]
    #                 if metric == 'mse':
    #                     print(rf'& ${row['Avg. MSE'].values[0]: .3f} \pm \text{{\scriptsize {row['Std. MSE'].values[0]: .3f}}}$', end=' ')
    #                 elif metric == 'acc':
    #                     if dataset == 'MetroTraffic':
    #                         print(rf'& ${row['Avg. hit acc.'].values[0]: .3f} \pm \text{{\scriptsize {row['Std. hit acc.'].values[0]: .3f}}}$',
    #                             end=' ')
    #
    #                     else:
    #                         print(
    #                             rf'& ${'-'}$',
    #                             end=' ')
    #         print('\\\\\n')
    # latex_file = r"""

    """PARALLELISM ABLATION"""
    df = pd.read_csv('experiments/ablations/parallelism/ablation_parallelism.csv')
    mapper = {"AR-16": "AR-16", "AR-32": "AR-32", "DNQ": "DNQ", "Pipeline": "Pipe", "AR-8": "AR-8"}
    for parallel in ["AR-8", "AR-16", "AR-32", "DNQ", "Pipeline"]:
        for dataset in ["PanamaEnergy"]:
            for level in ["C", "M", "F"]:
                row = df.loc[(df["Parallelism"] == parallel) & (df['Dataset'] == dataset) & (df['Level'] == level)]
                val = row["Avg. MSE"].values[0]
                std = row["Std. MSE"].values[0]
                queries = row["Queries"].values[0]
                time = row["Avg. Time"].values[0]
                std_time = row["Std. Time"].values[0]

                if level == "C":
                    print(
                        f" {mapper[parallel]} & ${val:.3f} \\pm \\text{{\\scriptsize{{{std:.3f}}}}}$ & ${int(queries)}$ & ${time:.3f} \\pm \\text{{\\scriptsize{{{std_time:.3f}}}}}$",
                        end="")
                elif level == "M":
                    print(
                        f" & ${val:.3f} \\pm \\text{{\\scriptsize{{{std:.3f}}}}}$ & ${int(queries)}$ & ${time:.3f} \\pm \\text{{\\scriptsize{{{std_time:.3f}}}}}$",
                        end="")
                else:
                    print(
                        f" & ${val:.3f} \\pm \\text{{\\scriptsize{{{std:.3f}}}}}$ & ${int(queries)}$ & ${time:.3f} \\pm \\text{{\\scriptsize{{{std_time:.3f}}}}}$\\\\")

import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv('experiments/ablations/encoding/ablation_encoding.csv')
    for encoding in ['OHE', 'ORD', 'STD', 'PROP']:
        for metric in ['mse', 'acc']:
            for dataset in ['AustraliaTourism', 'MetroTraffic']:
                for masking in ['C', 'M', 'F']:
                    row = df.loc[(df['Dataset'] == dataset) & (df['Mask'] == masking) & (df['Encoding'] == encoding)]
                    if metric == 'mse':
                        print(rf'& ${row['Avg. MSE'].values[0]: .3f} \pm \text{{\scriptsize {row['Std. MSE'].values[0]: .3f}}}$', end=' ')
                    elif metric == 'acc':
                        if dataset == 'MetroTraffic':
                            print(rf'& ${row['Avg. hit acc.'].values[0]: .3f} \pm \text{{\scriptsize {row['Std. hit acc.'].values[0]: .3f}}}$',
                                end=' ')

                        else:
                            print(
                                rf'& ${'-'}$',
                                end=' ')
            print('\\\\\n')
    latex_file = r"""
    \begin{table}[htb]
    \centering
    \caption{Comparison of Methods on Two Datasets with Different Masking Types and Metrics}
    \begin{tabular}{l ccc ccc}
        \toprule
        % Top row: Dataset headers
        & \multicolumn{3}{c}{\textbf{MetroTraffic}} & \multicolumn{3}{c}{\textbf{AustraliaTourism}} \\
        \cmidrule(lr){2-4} \cmidrule(lr){5-7}
        % Second row: Masking type headers for each dataset
        \textbf{Method} & \textbf{C} & \textbf{M} & \textbf{F} & \textbf{C} & \textbf{M} & \textbf{F} \\
        \midrule
        
        % Method 1 with two sub-rows for metrics
        \multirow{2}{*}{\textbf{OHE}} 
        & 0.90 & 0.85 & 0.87 & 0.88 & 0.91 & 0.89 \\  % Metric 1 row for Method 1
        & 0.85 & 0.80 & 0.83 & 0.82 & 0.86 & 0.84 \\  % Metric 2 row for Method 1
        \midrule
        
        % Method 2 with two sub-rows for metrics
        \multirow{2}{*}{\textbf{ORD}} 
        & 0.92 & 0.88 & 0.89 & 0.91 & 0.93 & 0.90 \\  % Metric 1 row for Method 2
        & 0.88 & 0.82 & 0.85 & 0.87 & 0.89 & 0.86 \\  % Metric 2 row for Method 2
        \midrule

        % Method 3 with two sub-rows for metrics
        \multirow{2}{*}{\textbf{STD}} 
        & 0.91 & 0.87 & 0.89 & 0.90 & 0.92 & 0.88 \\  % Metric 1 row for Method 3
        & 0.87 & 0.81 & 0.84 & 0.85 & 0.87 & 0.85 \\  % Metric 2 row for Method 3
        \midrule
        
        % Method 4 with two sub-rows for metrics
        \multirow{2}{*}{\textbf{PROP}} 
        & 0.93 & 0.89 & 0.90 & 0.92 & 0.94 & 0.91 \\  % Metric 1 row for Method 4
        & 0.89 & 0.83 & 0.86 & 0.88 & 0.90 & 0.87 \\  % Metric 2 row for Method 4
        
        \bottomrule
    \end{tabular}
    \label{tab:comparison}
\end{table}"""
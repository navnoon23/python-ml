import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stats

# load data
df = pd.read_csv('liverdata-sgdpred-50k-ptimeyr-120822-1.csv', header='infer')

# ptime yr bins
colname = 'PTIMEYR'
df1 = df[df[colname] < 1]
df2 = df[df[colname].between(1, 5, inclusive='left')]
df3 = df[df[colname].between(5, 10, inclusive='left')]
df4 = df[df[colname].between(10, 15, inclusive='left')]
df5 = df[df[colname].between(15, 20, inclusive='both')]
df6 = df[df[colname] > 20]

actual_counts = [
    df1.shape[0],
    df2.shape[0],
    df3.shape[0],
    df4.shape[0],
    df5.shape[0],
    df6.shape[0],
]

colname = 'PRED_PTYR'
dfpred1 = df[df[colname] < 1]
dfpred2 = df[df[colname].between(1, 5, inclusive='left')]
dfpred3 = df[df[colname].between(5, 10, inclusive='left')]
dfpred4 = df[df[colname].between(10, 15, inclusive='left')]
dfpred5 = df[df[colname].between(15, 20, inclusive='both')]
dfpred6 = df[df[colname] > 20]

predicted_counts = [
    dfpred1.shape[0],
    dfpred2.shape[0],
    dfpred3.shape[0],
    dfpred4.shape[0],
    dfpred5.shape[0],
    dfpred6.shape[0],
]

index_vals = [
    '<1',
    '1-5',
    '5-10',
    '10-15',
    '15-20',
    '>20',
]

plotdf = pd.DataFrame(
    {
        'Actual': actual_counts,
        'Predicted': predicted_counts,
     },
    index=index_vals
)

ax = plotdf.plot.bar(
    rot=0,
    title='Actual vs Predicted Survival in Years - Counts',
    xlabel='Survival in Years',
    ylabel='Count of Recipients',
    fontsize=12,
    color={
       'Actual': '#94c6eb',
       'Predicted': '#2d3361',
    }
)

for p in ax.patches:
    ax.annotate(str(np.round(p.get_height(), decimals=1)), (p.get_x() * 1.005, p.get_height() * 1.005))

plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stats

# load data
df = pd.read_csv('liverdata-sgdpred-50k-ptimeyr-120822-1.csv', header='infer')

# bins
colname = 'AgeDiff_P_D'
df1 = df[df[colname] < 5]
df2 = df[df[colname].between(5, 9, inclusive='both')]
df3 = df[df[colname].between(10, 14, inclusive='both')]
df4 = df[df[colname].between(15, 19, inclusive='both')]
df5 = df[df[colname].between(20, 24, inclusive='both')]
df6 = df[df[colname].between(25, 29, inclusive='both')]
df7 = df[df[colname] > 29]

colname = 'PTIMEYR'
ptimeyr_means = [
    stats.mean(df1[colname]),
    stats.mean(df2[colname]),
    stats.mean(df3[colname]),
    stats.mean(df4[colname]),
    stats.mean(df5[colname]),
    stats.mean(df6[colname]),
    stats.mean(df7[colname])
]

ptimeyr_stdev = [
    stats.stdev(df1[colname]),
    stats.stdev(df2[colname]),
    stats.stdev(df3[colname]),
    stats.stdev(df4[colname]),
    stats.stdev(df5[colname]),
    stats.stdev(df6[colname]),
    stats.stdev(df7[colname])
]

colname = 'PRED_PTYR'
pred_means = [
    stats.mean(df1[colname]),
    stats.mean(df2[colname]),
    stats.mean(df3[colname]),
    stats.mean(df4[colname]),
    stats.mean(df5[colname]),
    stats.mean(df6[colname]),
    stats.mean(df7[colname])
]

pred_stdev = [
    stats.stdev(df1[colname]),
    stats.stdev(df2[colname]),
    stats.stdev(df3[colname]),
    stats.stdev(df4[colname]),
    stats.stdev(df5[colname]),
    stats.stdev(df6[colname]),
    stats.stdev(df7[colname])
]

index_vals = [
    '<5 (' + '{:,}'.format(df1.shape[0]) + ')',
    '5-9 (' + '{:,}'.format(df2.shape[0]) + ')',
    '10-14 (' + '{:,}'.format(df3.shape[0]) + ')',
    '15-19 (' + '{:,}'.format(df4.shape[0]) + ')',
    '20-24 (' + '{:,}'.format(df5.shape[0]) + ')',
    '25-29 (' + '{:,}'.format(df6.shape[0]) + ')',
    '>29 (' + '{:,}'.format(df7.shape[0]) + ')'
]

plotdf = pd.DataFrame(
    {
        'Actual-Mean': ptimeyr_means,
        'Actual-StdDev': ptimeyr_stdev,
        'Predicted-Mean': pred_means,
        'Predicted-StdDev': pred_stdev
    },
    index=index_vals
)

ax = plotdf.plot.bar(
    rot=0,
    title='Actual vs Predicted Survival in Years - Recipient-Donor Age Difference',
    xlabel='Age Difference (#Recipients)',
    ylabel='Survival in Years',
    fontsize=12,
    color={
        'Actual-Mean': '#94c6eb',
        'Actual-StdDev': '#f99e1c',
        'Predicted-Mean': '#2d3361',
        'Predicted-StdDev': '#e56529'
    }
)

for p in ax.patches:
    ax.annotate(str(np.round(p.get_height(), decimals=1)), (p.get_x() * 1.005, p.get_height() * 1.005))

plt.show()
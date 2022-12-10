import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stats

# load data
df = pd.read_csv('liverdata-sgdpred-50k-ptimeyr-120822-1.csv', header='infer')

# pt age bins
colname = 'LOS'
df1 = df[df[colname] < 10]
df2 = df[df[colname].between(10, 30, inclusive='both')]
df3 = df[df[colname].between(31, 90, inclusive='both')]
df4 = df[df[colname].between(91, 180, inclusive='both')]
df5 = df[df[colname].between(181, 365, inclusive='both')]
df6 = df[df[colname].between(366, 730, inclusive='both')]
df7 = df[df[colname] > 731]

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
    '<10 (' + '{:,}'.format(df1.shape[0]) + ')',
    '10-30 (' + '{:,}'.format(df2.shape[0]) + ')',
    '31-90 (' + '{:,}'.format(df3.shape[0]) + ')',
    '91-180 (' + '{:,}'.format(df4.shape[0]) + ')',
    '181-365 (' + '{:,}'.format(df5.shape[0]) + ')',
    '366-730 (' + '{:,}'.format(df6.shape[0]) + ')',
    '<730 (' + '{:,}'.format(df7.shape[0]) + ')'
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
    title='Actual vs Predicted Survival in Years - Post Transplant Hospital Stay',
    xlabel='Recipient Hospital Stay in Days (#Recipients)',
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
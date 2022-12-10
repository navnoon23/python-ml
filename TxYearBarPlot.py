import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stats

# load data
df = pd.read_csv('liverdata-sgdpred-50k-ptimeyr-120822-1.csv', header='infer')

# bins
colname = 'TX_YEAR'
df1 = df[df[colname] < 1990]
df2 = df[df[colname].between(1990, 1994, inclusive='both')]
df3 = df[df[colname].between(1995, 1999, inclusive='both')]
df4 = df[df[colname].between(2000, 2004, inclusive='both')]
df5 = df[df[colname].between(2005, 2009, inclusive='both')]
df6 = df[df[colname].between(2010, 2014, inclusive='both')]
df7 = df[df[colname].between(2015, 2019, inclusive='both')]
df8 = df[df[colname] > 2019]

colname = 'PTIMEYR'
ptimeyr_means = [
    stats.mean(df1[colname]),
    stats.mean(df2[colname]),
    stats.mean(df3[colname]),
    stats.mean(df4[colname]),
    stats.mean(df5[colname]),
    stats.mean(df6[colname]),
    stats.mean(df7[colname]),
    stats.mean(df8[colname])
]

ptimeyr_stdev = [
    stats.stdev(df1[colname]),
    stats.stdev(df2[colname]),
    stats.stdev(df3[colname]),
    stats.stdev(df4[colname]),
    stats.stdev(df5[colname]),
    stats.stdev(df6[colname]),
    stats.stdev(df7[colname]),
    stats.stdev(df8[colname])
]

colname = 'PRED_PTYR'
pred_means = [
    stats.mean(df1[colname]),
    stats.mean(df2[colname]),
    stats.mean(df3[colname]),
    stats.mean(df4[colname]),
    stats.mean(df5[colname]),
    stats.mean(df6[colname]),
    stats.mean(df7[colname]),
    stats.mean(df8[colname])
]

pred_stdev = [
    stats.stdev(df1[colname]),
    stats.stdev(df2[colname]),
    stats.stdev(df3[colname]),
    stats.stdev(df4[colname]),
    stats.stdev(df5[colname]),
    stats.stdev(df6[colname]),
    stats.stdev(df7[colname]),
    stats.stdev(df8[colname])
]

index_vals = [
    '<1990 (' + '{:,}'.format(df1.shape[0]) + ')',
    '1990-94 (' + '{:,}'.format(df2.shape[0]) + ')',
    '1995-99 (' + '{:,}'.format(df3.shape[0]) + ')',
    '2000-04 (' + '{:,}'.format(df4.shape[0]) + ')',
    '2005-09 (' + '{:,}'.format(df5.shape[0]) + ')',
    '2010-14 (' + '{:,}'.format(df6.shape[0]) + ')',
    '2015-19 (' + '{:,}'.format(df7.shape[0]) + ')',
    '2020 (' + '{:,}'.format(df8.shape[0]) + ')'
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
    rot=15,
    title='Actual vs Predicted Survival in Years - Transplant Year',
    xlabel='Transplant Year Range (#Recipient)',
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
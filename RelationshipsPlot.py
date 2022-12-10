import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stats

# load data
df = pd.read_csv('liverdata-sgdpred-50k-ptimeyr-120822-1.csv', header='infer')

# pt age bins
df1 = df[df['AGE'] < 13]
df2 = df[df['AGE'].between(13, 19, inclusive='both')]
df3 = df[df['AGE'].between(20, 29, inclusive='both')]
df4 = df[df['AGE'].between(30, 39, inclusive='both')]
df5 = df[df['AGE'].between(40, 49, inclusive='both')]
df6 = df[df['AGE'].between(50, 59, inclusive='both')]
df7 = df[df['AGE'].between(60, 69, inclusive='both')]
df8 = df[df['AGE'] > 69]

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
    '0-12 ('+'{:,}'.format(df1.shape[0]) + ')',
    '13-19 ('+'{:,}'.format(df2.shape[0]) + ')',
    '20-29 ('+'{:,}'.format(df3.shape[0]) + ')',
    '30-39 ('+'{:,}'.format(df4.shape[0]) + ')',
    '40-49 ('+'{:,}'.format(df5.shape[0]) + ')',
    '50-59 ('+'{:,}'.format(df6.shape[0]) + ')',
    '60-69 ('+'{:,}'.format(df7.shape[0]) + ')',
    '70+ ('+'{:,}'.format(df8.shape[0]) + ')'
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
    title='Actual vs Predicted Survival in Years - Recipient Age',
    xlabel='Recipient Age (#Recipients)',
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
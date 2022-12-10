import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stats

# load data
df = pd.read_csv('liverdata-sgdpred-50k-ptimeyr-120822-1.csv', header='infer')

# bins
colname = 'LiveDon'
df1 = df[df[colname] <= 0]
df2 = df[df[colname] > 0]

colname = 'PTIMEYR'
ptimeyr_means = [
    stats.mean(df1[colname]),
    stats.mean(df2[colname])
]

ptimeyr_stdev = [
    stats.stdev(df1[colname]),
    stats.stdev(df2[colname])
]

colname = 'PRED_PTYR'
pred_means = [
    stats.mean(df1[colname]),
    stats.mean(df2[colname])
]

pred_stdev = [
    stats.stdev(df1[colname]),
    stats.stdev(df2[colname])
]

index_vals = [
    'Deceased Donor (' + '{:,}'.format(df1.shape[0]) + ')',
    'Live Donor (' + '{:,}'.format(df2.shape[0]) + ')'
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
    title='Actual vs Predicted Survival in Years - Live/Deceased Donor',
    xlabel='Live/Deceased Donor (#Recepients)',
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
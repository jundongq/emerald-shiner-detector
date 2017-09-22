import pandas as pd
import numpy as np 
import matplotlib.pylab as plt


# import the returned fish center coordinates by seeFish.py
# df has two rows with headers as 'cx' and 'cy'
df = pd.read_csv('Fish_1_0804_Converted/sampledFrames_MOG/fishCentroids.csv')

cols = df.columns.tolist()
print cols

threshold = 200
window    = 20

# obtain medians for elements falling in the rolling window
for col in cols:
	df['{}_median'.format(col)] = df['{}'.format(col)].rolling(window=window, center=True).\
	median().fillna(method='bfill').fillna(method='ffill')


diff_cx = np.abs(df['{}'.format('cx')] - df['{}_median'.format('cx')])
diff_cy = np.abs(df['{}'.format('cy')] - df['{}_median'.format('cy')])

outlier_cx = diff_cx > threshold
outlier_cy = diff_cy > threshold


outlier_cx_idx = df['cx'][outlier_cx].index.tolist()
outlier_cy_idx = df['cy'][outlier_cy].index.tolist()

outlier_idx = sorted(list(set(np.concatenate((outlier_cx_idx, outlier_cy_idx)))))
print outlier_idx


from operator import itemgetter
from itertools import groupby

# group the outlier idices by continuity

ranges = []
for k, g in groupby(enumerate(outlier_idx), lambda (i,x):i-x):
    group = map(itemgetter(1), g)
    ranges.append((group[:]))

print ranges

df['cx_smooth'] =  df['cx'].copy()
df['cy_smooth'] =  df['cy'].copy()
for i, l in enumerate(ranges):

	# fill the outlier space with values equally seperated by the previous one and trailling one
    df['cx_smooth'].iloc[l] = [np.int(x) for x in np.linspace(df['cx'][l[0]-1], df['cx'][l[-1]+1], len(l)+2)[1:-1]]
    df['cy_smooth'].iloc[l] = [np.int(x) for x in np.linspace(df['cy'][l[0]-1], df['cy'][l[-1]+1], len(l)+2)[1:-1]]

df.to_csv('Fish_1_0804_Converted/sampledFrames_MOG/fishCentroids_smooth.csv')








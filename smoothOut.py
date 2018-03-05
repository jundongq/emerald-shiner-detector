import pandas as pd
import numpy as np
import argparse 
import os
import matplotlib.pylab as plt


# import the returned fish center coordinates by seeFish.py
# df has two rows with headers as 'cx' and 'cy'

def smooth_fishCentroid(dir, score_threshold, smooth_threshold, window):
	"""
	Input::
	
	dir:
	a directory that contains all related images. For example 'FishBehavior_11_09_9' is 
	a directory containing all imgs of 'FishBehavior_11_09_9.mp4'.
	
	smooth_threshold:
	if a data point's distance to the median value of a rolling window exceed the threshold
	it is considered as an outlier, so it needs to be smoothed. Threshold is in pixel unit.
	
	window:
	number of elements taken into acount while computing median.
	
	Output::
	a csv file containing fish mass center in pixel unit. 'fishCentroids_smooth.csv'
	"""
	
	csv_file = os.path.join(dir, 'fishCentroids_{}.csv'.format(score_threshold))
	print csv_file
	df = pd.read_csv(csv_file)

	cols = df.columns.tolist()
	print cols

	# obtain medians for elements falling in the rolling window
	for col in cols:
		df['{}_median'.format(col)] = df['{}'.format(col)].rolling(window=window, center=True).\
		median().fillna(method='bfill').fillna(method='ffill')


	diff_cx = np.abs(df['{}'.format('cx')] - df['{}_median'.format('cx')])
	diff_cy = np.abs(df['{}'.format('cy')] - df['{}_median'.format('cy')])

	outlier_cx = diff_cx > smooth_threshold
	outlier_cy = diff_cy > smooth_threshold


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
		if l[0] == 0:
			starting_ = l[0]
		else:
			starting_ = l[0]-1

		if l[-1] == df.shape[0]-1:
			ending_ = df.shape[0]-1
		else: 
			ending_ = l[-1]+1
		# fill the outlier space with values equally seperated by the previous one and trailling one
		df['cx_smooth'].iloc[l] = [np.int(x) for x in np.linspace(df['cx'][starting_], df['cx'][ending_], len(l)+2)[1:-1]]
		df['cy_smooth'].iloc[l] = [np.int(x) for x in np.linspace(df['cy'][starting_], df['cy'][ending_], len(l)+2)[1:-1]]


	df.to_csv(os.path.join(dir, 'fishCentroids_{}_smooth.csv'.format(score_threshold)))

if __name__ == '__main__':

	ap = argparse.ArgumentParser()
	ap.add_argument('-d', '--dir', required=True, help='dirctory that contains fishCentroids.csv')
	ap.add_argument('-s', '--score_threshold', type=float, required=True, help='threshold to draw interrogation masks')
	ap.add_argument('-t', '--smooth_threshold', type=int, required=True, help='threshold to smooth out outliers')
	ap.add_argument('-w', '--window', type=int, required=True, help='window size to compute rolling median')
	args = vars(ap.parse_args())

	data_dir         = args['dir']
	score_threshold  = args['score_threshold']
	smooth_threshold = args['smooth_threshold']
	window           = args['window']
	
	smooth_fishCentroid(data_dir, score_threshold, smooth_threshold, window)






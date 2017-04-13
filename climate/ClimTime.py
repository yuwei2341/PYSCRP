''' Module for time operation in climate analysis '''
import numpy as np

def MonAve(data, mstart, mend):
	"""
	MonAve - average from the mstart month to the mend month of each year.
	e.g, for JJA, mstart=6, mend=8; for DJF, mstart=12, mend=14,
	and the average of the last year is invalid
	"""
	sz_data = data.shape
	ny = np.int(np.floor(data.shape[0] / 12)) # number of years
	month = np.arange(mstart, mend + 1)

	# whether the time lasts to the next year
	mextra = mend - 12
	if mextra > 0:
		tmp = np.empty((mextra, ) + sz_data[1:])
		tmp[:] = np.NAN
		data = np.concatenate((data, tmp), axis = 0)

	dataMonAve = np.zeros((ny, ) + sz_data[1:])


	if data.ndim == 3:
		for ii in np.arange(ny):
			count = 12 * ii + month - 1
			dataMonAve[ii, :, :] = np.nanmean(data[count, :, :], axis = 0)
	elif data.ndim == 2:
		for ii in np.arange(ny):
			count = 12 * ii + month - 1
			dataMonAve[ii, :] = np.nanmean(data[count, :], axis = 0)
	elif data.ndim == 1:
		for ii in np.arange(ny):
			count = 12 * ii + month - 1
			dataMonAve[ii] = np.nanmean(data[count], axis = 0)
	else:
		print 'wrong dims'
		return
	
	return dataMonAve

	
	

def MovingAve(a, n = 3):
    """
    MovingAve(a, n = 3) computes the moving average of an index
        a: a one dimensional array
        n: time window for moving average
    Returns: a one dimensional array that has been applied moving average on
    """
    ret = np.cumsum(a, dtype = float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
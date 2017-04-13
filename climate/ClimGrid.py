"""
This module is for commonly used climate analysis
"""
import numpy as np
from mpl_toolkits.basemap import shiftgrid, maskoceans

def AreaMean(data, sData):
	""" 
	AreaMeanStr -- Area-weighted average over a certain area
	   data: the 3D [time, lat, lon] to be averaged
	   sData: a dictionary of info of data: lat, lon, slat, elat, slon, elon
	"""         
	# Get indices of lat/lon WITHIN the desiganated region
	x1 = np.where(sData['lon'] >= sData['slon'])[0][0]
	x2 = np.where(sData['lon'] <= sData['elon'])[0][-1]
	y1 = np.where(sData['lat'] >= sData['slat'])[0][0]
	y2 = np.where(sData['lat'] <= sData['elat'])[0][-1]

	lonRg = sData['lon'][x1:(x2 + 1)]
	latRg = sData['lat'][y1:(y2 + 1)];

	# get regional data
	dataRegion = data[:, y1:(y2 + 1), x1:(x2 + 1)];

	# weight by latitude
	dataWt = LatWeight(dataRegion, latRg, 'sphere')

	# Get regional mean
	dataWtMean = np.nansum(dataWt, axis = (1, 2))
	return dataWtMean, lonRg, latRg
	

def LatWeight(data, lat, wtType):
	""" 
	LatWeight -- Weight by cos(latitude)
	data: input data to be weighted
	lat: latitude of the data
	wtType: method of weighting.
		For 'sqrt': just multiply each grid by its sqrt(cos(lat)), the use of
		this is to prepare data for covariance matrix for EOF
		For 'sphere': multiply each grid weighted by cos(lat), and then divided 
		by the sum of the weighting (the effective weighting, meaning not taking 
		into account of the grid with NaN). To obtain area-weighted mean then,
		simply sum up the result
		
	viva-141104: 
			tmpMask = ~np.isnan(tmp)
			sumWt = (sqcoslat * tmpMask).sum()
		moved inside if wtType == 'sphere':
		and changed to
			tmpMask = ~np.isnan(tmp)
			sumWt = np.nansum(sqcoslat * tmpMask)
	"""

	P = np.shape(data)

	dataWt = np.zeros(P)

	if wtType == 'sqrt':
		sqcoslat = np.transpose(np.tile(np.sqrt(np.cos(lat * np.pi / 180)), (P[2], 1)))
	elif wtType == 'sphere':
		sqcoslat = np.transpose(np.tile(np.cos(lat * np.pi / 180), (P[2], 1)))
	else:
		print('wrong wtType')

	# weight
	for it in np.arange(P[0]):
		tmp = data[it, :, :]
		dataWt[it, :, :] = tmp * sqcoslat
		if wtType == 'sphere':
			tmpMask = ~np.isnan(tmp)
			sumWt = np.nansum(sqcoslat * tmpMask)
			dataWt[it, :, :] = dataWt[it, :, :] / sumWt

	return dataWt


def TopoMask(lat, lon, isZeroMid, maskType):
	"""
	TopoMask -- Derive mask data for a specific lat and lon
	lat, lon: the latitude and longitude
	isZeroMid: whether 0 longitude is in the middle; otherwise, it's at the start
	maskType: mask land or ocean
	"""

	topodatin = np.zeros((lat.size, lon.size))

	# shift data so lons go from -180 to 180 if necessary
	if isZeroMid:
		topoin = topodatin
		lons1 = lon
	else:
		topoin, lons1 = shiftgrid(180., topodatin, lon, start = False)

	lats1 = lat
	lons, lats = np.meshgrid(lons1, lats1)

	# interpolate land/sea mask to topo grid, mask ocean values.
	topo = maskoceans(lons, lats, topoin)

	# convert masked array to np array and shift 1 and 0 so as to mask land
	if maskType == 'land':
		topo = np.ma.filled(topo, 1)
		topo[topo == 0] = np.nan
	elif maskType == 'ocean':
		topo = np.ma.filled(topo, np.nan)
		topo[topo == 0] = 1

	# If data are not from -180 to 180
	if not isZeroMid:
		topo, tmp = shiftgrid(0, topo, lonsin, start = True)    
	return topo
	
def DeGlbMean(data, lat, method):
    """
    DeGlbMean -- Returns data[time, lat, lon] with global mean subtracted, and the global mean time series.
    data: input data, must be [time, lat, lon]
    lat: latitude of the data
    method: methods of de-global-mean:
        'PointTrend': get the trend of each point, the subtract it for each point
        'Origin': de global mean
        'Trend': get the trend in global mean, then subtract it from data;
    viva-141104: only finished the Origin method
    """
    P = np.shape(data);
    mtmp1 = LatWeight(data, lat, 'sphere');
    # mtmp1 here alreday divided by the weight, so simply summing up will give global mean
    mglb = np.nansum(mtmp1, axis = (1, 2))
    mmglb = np.transpose(np.tile(mglb, (P[1], P[2], 1)), (2, 0, 1))
    return data - mmglb, mglb
    
    
def GetRegion(sEOFIn, data, lat, lon):
    """
    GetRegion(sEOFIn, data, lon, lat) -- Select fields of designated region.
    The domain is WITHIN (not out of) the specified region.
    sEOFIn: dict containing region parameters: slon, elon, slat and elat
    data, lon, lat: field (data) of three dimensions (time, lat, lon), and
    longitude and latitude of the field
    dataRg, lonRg, latRg: data (dataRg), longitude (lonRg) and
    latitude (lonRg) of the designated region appended
    """
    
    ## Get indices of lat/lon WITHIN the desiganated region
    x1 = np.where(lon >= sEOFIn['slon'])[0][0]
    x2 = np.where(lon <= sEOFIn['elon'])[0][-1]
    y1 = np.where(lat >= sEOFIn['slat'])[0][0]
    y2 = np.where(lat <= sEOFIn['elat'])[0][-1]
    lonRg = lon[x1:(x2 + 1)]
    latRg = lat[y1:(y2 + 1)]
    
    ## Get the region
    dataRg = data[:, y1:(y2 + 1), x1:(x2 + 1)]

    return dataRg, latRg, lonRg
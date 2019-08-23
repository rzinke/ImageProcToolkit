import numpy as np 
import matplotlib.pyplot as plt 
from osgeo import gdal
from fitPlane import *


##############################
### --- GPS formatting --- ###
##############################

# --- Read in GPS data from file ---
class readGPSdata:
	'''
	'''
	def __init__(self,fname,skiplines,xCol,yCol,Ecol=None,Ncol=None,Zcol=None,siteCol=None,delimiter=' ',vocal=False):
		# Open text file and read in lines
		Fin=open(fname,'r')
		Lines=Fin.readlines()
		Fin.close()
		# Parse data
		Lines=Lines[skiplines:] # skip header
		nLines=len(Lines) # number of lines
		if vocal is True:
			print('Reading in data...\n\tFile: {}\n\tReading {} lines'.format(fname,nLines))
		self.x=np.zeros(nLines) # Lon/easting
		self.y=np.zeros(nLines) # Lat/northing
		if Ecol: # empty east column
			self.E=np.zeros(nLines)
			if vocal is True:
				print('\tE component: Col. {}'.format(Ecol))
		if Ncol: # empty north column
			self.N=np.zeros(nLines)
			if vocal is True:
				print('\tN component: Col. {}'.format(Ncol))
		if Zcol: # empty vertical column
			self.Z=np.zeros(nLines)
			if vocal is True:
				print('\tZ component: Col. {}'.format(Zcol))
		if siteCol is not None:
			self.site=[]
			if vocal is True: 
				print('\tSite specified: Col. {}'.format(siteCol))
		for i in range(nLines):
			line=Lines[i].split(delimiter)
			self.x[i]=float(line[xCol]) # Lon/easting
			self.y[i]=float(line[yCol]) # Lat/northing
			if Ecol:
				self.E[i]=line[Ecol]
			if Ncol:
				self.N[i]=line[Ncol]
			if Zcol:
				self.Z[i]=line[Zcol]
			if siteCol is not None:
				self.site.append(line[siteCol])


# --- GPS points object ---
class GPSobj:
	'''
	'''
	def __init__(self,x,y,E=None,N=None,Z=None,site=None,inputSRS='EPSG:4326',outputSRS='EPSG:4326'):
		print('Creating GPS object')

		# Record coordinates
		self.x=x
		self.y=y

		# Displacements
		self.displacements={} # empty dictionary
		if E is not None:
			self.displacements['E']=E
		if N is not None:
			self.displacements['N']=N
		if Z is not None:
			self.displacements['Z']=Z
		try:
			self.displacements['Unet']=np.sqrt(E**2+N**2+Z**2)
		except:
			pass

		# Convert to desired coordinates
		if outputSRS != inputSRS:
			print('\tChanging GPS coordinates system \n\tfrom: %s\n\tto: %s' % (inputSRS,outputSRS))
			from pyproj import Proj,transform # import library
			inProj = Proj(init=inputSRS)
			outProj = Proj(init=outputSRS)
			self.x,self.y = transform(inProj,outProj,self.x,self.y)

	# Plot if desired
	def plot(self,value=None,title=None):
		F=plt.figure()
		ax=F.add_subplot(111)
		ax.scatter(self.x,self.y,color='k')
		if value is not None:
			cax=ax.scatter(self.x,self.y,c=value)
			F.colorbar(cax,orientation='vertical')
		ax.set_aspect(1)
		ax.set_title(title)



##################################
### --- GPS-map comparison --- ###
##################################

# --- Displacement map vs GPS comparison ---
class gpsCompare:
	'''
	'''
	## Compute differences
	def __init__(self,ImgObj,GPSobj,imgBand=1,gpsComponent='E',xWidth=3,yWidth=3,expc='median'):
		print('Comparing displacement field to GPS')
		# Extract parameters from ImgObj
		try: # test if gdal object
			img=ImgObj.GetRasterBand(imgBand).ReadAsArray()
		except: # if raster image
			img=ImgObj
			print('No spatial info recorded')

		# Reference frame
		m=img.shape[0]; n=img.shape[1]
		try: 
			transform=ImgObj.GetGeoTransform()
			dx=transform[1]; dy=transform[5]
			left=transform[0]; right=left+n*dx
			top=transform[3]; bottom=top+m*dy
			extent=(left, right, bottom, top) # left right bottom top
		except:
			top=0; bottom=m 
			left=0; right=n 
			extent=(left, right, bottom, top) 
		print('\tImage extent:',extent)

		# Convert GPSobj into 3-col array [x, y , u]
		try: # test if gps object defined above
			Ngps=len(GPSobj.x) # number of data points
			gps=np.zeros((Ngps,3)) # empty array
			gps[:,0]=GPSobj.x # x position
			gps[:,1]=GPSobj.y # y position
			gps[:,2]=GPSobj.displacements[gpsComponent] # component of interest
		except:
			if type(GPSobj) is np.array:
				print('GPS already in array format: [x, y, u]')
			else:
				print('GPS not properly formatted -- Use GPSobj')

		# Crop GPS to relevant extent
		w=(gps[:,0]>=left)&(gps[:,0]<=right) # limit E-W
		gps=gps[w,:]
		w=(gps[:,1]>=bottom)&(gps[:,1]<=top) # limit N-S
		gps=gps[w,:]
		Ngps=gps.shape[0] # update number of stations
		print('\tGPS data cropped to map extent: %i points' % (Ngps))

		# Global min max displacement values
		print('\tSetting color scale')
		imgMin=np.nanmin(img); imgMax=np.nanmax(img)
		print('\timgMin: %f\timgMax: %f' % (imgMin,imgMax))
		gpsMin=np.nanmin(gps[:,2]); gpsMax=np.nanmax(gps[:,2])
		print('\tgpsMin: %f\tgpsMax: %f' % (gpsMin,gpsMax))
		globalMin=min((imgMin,gpsMin)); globalMax=max((imgMax,gpsMax))
		print('\tglobal Min: %f' % (globalMin))
		print('\tglobal Max: %f' % (globalMax))

		# Map area around points and difference
		expc=expc.lower() # define expectation operator
		if expc=='mean':
			E=np.nanmean
		elif expc=='median':
			E=np.nanmedian
		xWidth2=int(xWidth/2) # half kernel width for x range
		yWidth2=int(yWidth/2) # half kernel width for y range
		MapVal=np.zeros((Ngps,1)) # empty array for map values
		Diff=np.zeros((Ngps,1)) # empty array for GPS - map differences
		print('\tImage size: %i x %i' % (m,n))
		for i in range(Ngps):
			# Find row and column indices for each location
			Locx=gps[i,0]; colNdx=int((Locx-left)/dx)
			Locy=gps[i,1]; rowNdx=int((Locy-top)/dy)
			colRange=(colNdx-xWidth2,colNdx+xWidth2+1)
			rowRange=(rowNdx-yWidth2,rowNdx+yWidth2+1)

			# Find expected value within each kernel
			MapCell=img[rowRange[0]:rowRange[1],colRange[0]:colRange[1]]
			MapVal[i,0]=E(MapCell) # expected value

			# Diff = GPS - map
			Diff[i,0]=gps[i,2]-MapVal[i,0]

		# Concatenate difference
		outputMatrix=np.hstack((gps,MapVal,Diff))
		w=(np.isnan(Diff.squeeze(1))==0)
		outputMatrix=outputMatrix[w,:] # remove NaNs
		Ngps=outputMatrix.shape[0]

		# Store values for output
		self.img=img; img=None # image
		self.extent=extent # geographic extent
		self.Ngps=Ngps # number of data points
		self.gps=gps; gps=None
		self.easting=outputMatrix[:,0]
		self.northing=outputMatrix[:,1]
		self.gps_values=outputMatrix[:,2]
		self.map_values=outputMatrix[:,3]
		self.differences=outputMatrix[:,4]
		self.vmin=globalMin # color values
		self.vmax=globalMax

	## Save to files
	def writeToFile(self,basename):
		meanDiff=np.mean(self.differences)
		medianDiff=np.median(self.differences)
		absMeanDiff=np.mean(np.abs(self.differences))
		absMedianDiff=np.median(np.abs(self.differences))
		header='# GPS - map:\
		\n# mean diff: %f\tmedian diff: %f\
		\n# abs mean diff: %f\tabs median diff: %f\
		\n# easting, northing, gps (m), map (m), gps-map diff (m)\n'
		F=open(basename+'_output.txt','w')
		F.write(header % (meanDiff,medianDiff,absMeanDiff,absMedianDiff))
		for i in range(self.Ngps):
			F.write('%f %f %f %f %f\n' % (self.easting[i],self.northing[i],self.gps_values[i],
				self.map_values[i],self.differences[i]))
		F.close()

	## Plot
	def plot(self,title=None,cmap='viridis',basename=None):
		# Plot values
		F=plt.figure()
		ax=F.add_subplot(111)
		# Plot displacement map
		cax=ax.imshow(self.img,cmap=cmap,extent=self.extent,
			vmin=self.vmin,vmax=self.vmax,zorder=1)
		F.colorbar(cax,orientation='vertical')
		# Plot GPS points
		ax.scatter(self.easting,self.northing,c=self.gps_values,
			cmap=cmap,edgecolors='k',vmin=self.vmin,vmax=self.vmax,zorder=2)
		ax.set_aspect(1)
		if title:
			ax.set_title(title+' values')
		if basename is not None:
			F.savefig(basename+'_values.png',dpi=300)

		# Color range for difference values
		diffmax=np.max(np.abs(self.differences))
		diffmin=-diffmax
		# Plot differences
		F=plt.figure()
		ax=F.add_subplot(111)
		# Plot displacement map
		ax.imshow(self.img,cmap=cmap,extent=self.extent,
			vmin=self.vmin,vmax=self.vmax,zorder=1)
		# Plot GPS differences
		cax=ax.scatter(self.easting,self.northing,c=self.differences,
			cmap='jet',edgecolors='k',vmin=diffmin,vmax=diffmax,zorder=2)
		F.colorbar(cax,orientation='vertical')
		ax.set_aspect(1)
		if title:
			ax.set_title(title+' differences (GPS-map values)')
		if basename is not None:
			F.savefig(basename+'_differences.png',dpi=300)



# --- Adjust displacement map to GPS offsets ---
class gpsAdjust:
	def __init__(self,ImgObj,GPSobj,imgBand=1,gpsComponent='E',xWidth=3,yWidth=3,expc='median',vocal=False,plotBaseName=None):
		'''
		'''
		## Compute difference
		print('Initiating GPS adjust...')
		Comparison=gpsCompare(ImgObj,GPSobj,imgBand=imgBand,gpsComponent=gpsComponent,xWidth=xWidth,yWidth=yWidth,expc=expc)
		# List of available attributes
		# img = image
		# extent = geographic extent
		# Ngps = number of data points
		# gps = all gps data
		# easting = relevant gps easting
		# northing = relevant gps northing
		# gps_values = relevant gps values
		# map_values = map values at relevant gps locations
		# differences = GPS - map differences
		# vmin = global min displacement value
		# vmax = global max displacement value

		mImg,nImg=Comparison.img.shape
		left=Comparison.extent[0]; right=Comparison.extent[1]
		bottom=Comparison.extent[2]; top=Comparison.extent[3]

		# Difference stats
		meanDiff=np.nanmean(Comparison.differences)
		meanDiffOrig=np.nanmean(Comparison.differences)
		meanAbsDiffOrig=np.nanmean(np.abs(Comparison.differences))
		print('Uncorrected mean difference: {}'.format(meanDiff))

		# Plot difference data
		if plotBaseName is not None:
			# Color range for difference values
			diffmax=np.max(np.abs(Comparison.differences))
			diffmin=-diffmax	
			# Plot input and difference data
			F=plt.figure()
			ax=F.add_subplot(111)
			ax.imshow(Comparison.img,cmap='viridis',extent=Comparison.extent,
				vmin=Comparison.vmin,vmax=Comparison.vmax,zorder=1)
			# Plot GPS differences
			cax=ax.scatter(Comparison.easting,Comparison.northing,c=Comparison.differences,
				cmap='jet',edgecolors='k',vmin=diffmin,vmax=diffmax,zorder=2)
			F.colorbar(cax,orientation='vertical')
			ax.set_aspect(1)
			ax.set_title('Differences (GPS-map values)')
			F.savefig(plotBaseName+'_differences.png',dpi=300)

		# Subtract mean from map
		Comparison.img+=meanDiff # add mean difference to image
		Comparison.map_values+=meanDiff # add mean diff to relevant map values
		Comparison.differences=Comparison.gps_values-Comparison.map_values # recompute difference
		meanDiff=np.nanmean(Comparison.differences) # recompute adjusted mean
		print('Removed constant offset. New mean diff: {}'.format(meanDiff))

		## Fit plane to differences
		# Format mx3 matrix with [x,y,difference] columns
		diffData=np.hstack((Comparison.easting.reshape(-1,1),
			Comparison.northing.reshape(-1,1),
			Comparison.differences.reshape(-1,1)))
		# Remove center of data
		xCenter=np.mean(Comparison.easting) # center x location
		yCenter=np.mean(Comparison.northing) # center y location
		diffData[:,0]=diffData[:,0]-xCenter # remove x center
		diffData[:,1]=diffData[:,1]-yCenter # remove y center
		# Solve for plane
		if plotBaseName is not None:
			plot=True # specify plotting True/False
		DiffPlane=fitPlane(diffData,'pts',ds=0,includeAll=True,vocal=vocal,plot=True)

		# Construct plane at image resolution
		xPlane=np.linspace(left-xCenter,right-xCenter,nImg) # x locations
		yPlane=np.linspace(bottom-yCenter,top-yCenter,mImg) # y locations
		Xplane,Yplane=np.meshgrid(xPlane,yPlane) # coordinate grids
		Diff=-1/DiffPlane.N[2]*(DiffPlane.N[0]*Xplane+DiffPlane.N[1]*Yplane)
		Diff=np.flipud(Diff)

		if vocal is True:
			print('Constructing difference plane')
			print('\tImage size: {}x{}'.format(mImg,nImg))
			print('\tLeft: {}; Right: {}; Bottom: {}; Top: {}'.format(left,right,bottom,top))
		
		if plot is True:
			# Plot difference plane
			F=plt.figure()
			ax=F.add_subplot(111)
			cax=ax.imshow(Diff,cmap='viridis')#,#vmin=Comparison.vmin,vmax=Comparison.vmax,
				#extent=(left-xCenter,right-xCenter,bottom-yCenter,top-yCenter))
			F.colorbar(cax,orientation='horizontal')

		# Remove difference from fit image
		print('Removing difference plane')
		imgCrt=Comparison.img+Diff # crt is "corrected"

		if plotBaseName is not None:
			# Color range for difference values
			diffmax=np.max(np.abs(Comparison.differences))
			diffmin=-diffmax	
			# Plot input and difference data
			F=plt.figure()
			ax=F.add_subplot(111)
			cax=ax.imshow(imgCrt,cmap='viridis',extent=Comparison.extent)
			F.colorbar(cax,orientation='vertical')
			ax.set_aspect(1)
			ax.set_title('Corrected map')

		## Save corrected image
		if plotBaseName is not False:
			outName='{}_corrected.tif'.format(plotBaseName)
			print('... Saving to: {}'.format(plotBaseName))
			driver=gdal.GetDriverByName('GTiff') 
			CorrectedImage=driver.Create(outName,ImgObj.RasterXSize,ImgObj.RasterYSize,1,gdal.GDT_Float32) 
			CorrectedImage.GetRasterBand(1).WriteArray(imgCrt) 
			CorrectedImage.SetProjection(ImgObj.GetProjection()) 
			CorrectedImage.SetGeoTransform(ImgObj.GetGeoTransform()) 
			CorrectedImage.FlushCache() 
			print('Saved!')

			## Calculate and resport corrected stats
			ComparisonCrt=gpsCompare(CorrectedImage,GPSobj,imgBand=imgBand,gpsComponent=gpsComponent,
				xWidth=xWidth,yWidth=yWidth,expc=expc)
			meanDiffCrt=np.nanmean(ComparisonCrt.differences)
			meanAbsDiffCrt=np.nanmean(np.abs(ComparisonCrt.differences))
			print('--> Orig. mean difference: {}'.format(meanDiffOrig))
			print('--> Orig. mean abs difference: {}'.format(meanAbsDiffOrig))
			print('--> Final mean difference: {}'.format(meanDiffCrt))
			print('--> Final mean abs difference: {}'.format(meanAbsDiffCrt))		



###########################################
### --- Adjust images to each other --- ###
###########################################

# --- Fit one image to another image ---
class cinchImages:
	'''
	Base image should be coarser
	'''
	def __init__(self,baseImg_path,fitImg_path,upFactor=1,svdDnsmp='auto',saveImg=True,vocal=False,plot=False):
		# Remove file extension
		baseName=baseImg_path.split('.')[0] # base name sans extension
		fitName=fitImg_path.split('.')[0] # fit name sans extension

		# Load base image
		BaseDS=gdal.Open(baseImg_path,gdal.GA_ReadOnly)
		baseT=BaseDS.GetGeoTransform()

		# Upsample base image
		if upFactor != 1:
			newXres=baseT[1]/upFactor
			newYres=baseT[5]/upFactor
			BaseDS=gdal.Warp('',BaseDS,options=gdal.WarpOptions(
				format='MEM',xRes=newXres,yRes=newYres,
				resampleAlg='lanczos'))
			print('! Upsampled base image by {}: New pixel ({}x{})'.format(upFactor,newXres,newYres))
			baseT=BaseDS.GetGeoTransform() # recompute transform

		# Load fit image
		FitDS=gdal.Open(fitImg_path,gdal.GA_ReadOnly)
		fitT=FitDS.GetGeoTransform()

		# Geographic extent
		baseN=BaseDS.RasterXSize; baseM=BaseDS.RasterYSize
		baseXstep=baseT[1]; baseYstep=baseT[5]
		baseLeft=baseT[0]
		baseRight=baseLeft+baseXstep*baseN
		baseTop=baseT[3]
		baseBottom=baseTop+baseYstep*baseM
		baseExtent=(baseLeft,baseRight,baseBottom,baseTop)

		fitN=FitDS.RasterXSize; fitM=FitDS.RasterYSize
		fitXstep=fitT[1]; fitYstep=fitT[5]
		fitLeft=fitT[0]
		fitRight=fitLeft+fitXstep*fitN 
		fitTop=fitT[3]
		fitBottom=fitTop+fitYstep*fitM
		fitExtent=(fitLeft,fitRight,fitBottom,fitTop)

		if vocal is True:
			print('Base image: {}\n\t({}x{}) {}'.format(baseImg_path,baseM,baseN,baseT))
			print('Fit image: {}\n\t({}x{}) {}'.format(fitImg_path,fitM,fitN,fitT))

		if plot is True:
			baseImg=BaseDS.GetRasterBand(1).ReadAsArray()
			fitImg=FitDS.GetRasterBand(1).ReadAsArray()
			vmin=np.nanmin(fitImg); vmax=np.nanmax(fitImg)

			F=plt.figure()
			axBase=F.add_subplot(121)
			caxBase=axBase.imshow(baseImg,cmap='viridis',
				vmin=vmin,vmax=vmax,extent=baseExtent)
			axBase.set_title('Base')
			F.colorbar(caxBase,orientation='horizontal')
			axFit=F.add_subplot(122)
			caxFit=axFit.imshow(fitImg,cmap='viridis',
				vmin=vmin,vmax=vmax,extent=fitExtent)
			axFit.set_title('Fit')
			F.colorbar(caxFit,orientation='horizontal')

		# Resample fit image to base image resolution
		FitDSR=gdal.Warp('{}_downsampled.VRT'.format(fitName),FitDS,options=gdal.WarpOptions(
			format='VRT',xRes=baseXstep,yRes=baseYstep,
			resampleAlg='lanczos'))

		fitTR=FitDSR.GetGeoTransform()
		fitNR=FitDSR.RasterXSize; fitMR=FitDSR.RasterYSize
		fitXstepR=fitTR[1]; fitYstepR=fitTR[5]
		fitLeftR=fitTR[0]
		fitRightR=fitLeftR+fitXstepR*fitNR
		fitTopR=fitTR[3]
		fitBottomR=fitTopR+fitYstepR*fitMR
		fitExtentR=(fitLeftR,fitRightR,fitBottomR,fitTopR)

		if vocal is True:
			print('Resampled Fit to xRes: {}; yRes: {}\n\t{}'.format(fitXstepR,fitYstepR,fitTR))

		# Crop base image to fit boundaries
		bounds=(fitLeftR,fitBottomR,fitRightR,fitTopR)
		BaseDSR=gdal.Warp('{}_downsampled.VRT'.format(baseName),BaseDS,options=gdal.WarpOptions(
			format='VRT',outputBounds=bounds))

		baseTR=BaseDSR.GetGeoTransform()
		baseNR=BaseDSR.RasterXSize; baseMR=BaseDSR.RasterYSize
		baseXstepR=baseTR[1]; baseYstepR=baseTR[5]
		baseLeftR=baseTR[0]
		baseRightR=baseLeftR+baseXstepR*baseNR
		baseTopR=baseTR[3]
		baseBottomR=baseTopR+baseYstepR*baseMR
		baseExtentR=(baseLeftR,baseRightR,baseBottomR,baseTopR)

		if vocal is True:
			print('Cropped Base to (xmin ymin xmax ymax): {}\n\t{}'.format(bounds,baseTR))

		# Ensure coregistration
		assert baseExtentR==fitExtentR, '--- Images not properly aligned'
		ExtentR=baseExtentR

		# Difference Base - Fit
		fitImgR=FitDSR.GetRasterBand(1).ReadAsArray() # load as numpy arrays
		baseImgR=BaseDSR.GetRasterBand(1).ReadAsArray()
		Diff=baseImgR-fitImgR # take difference

		meanDiff=np.nanmean(Diff)
		meanDiffOrig=np.nanmean(Diff)
		meanAbsDiffOrig=np.nanmean(np.abs(Diff))
		print('Uncorrected mean difference: {}'.format(meanDiff))

		if plot is True:
			baseImgR=BaseDSR.GetRasterBand(1).ReadAsArray()
			fitImgR=FitDSR.GetRasterBand(1).ReadAsArray()
			vmin=np.nanmin(fitImgR); vmax=np.nanmax(fitImgR)

			F=plt.figure()
			axBase=F.add_subplot(131)
			caxBase=axBase.imshow(baseImgR,cmap='viridis',
				vmin=vmin,vmax=vmax,extent=baseExtentR)
			axBase.set_title('Base')
			F.colorbar(caxBase,orientation='horizontal')
			axFit=F.add_subplot(132)
			caxFit=axFit.imshow(fitImgR,cmap='viridis',
				vmin=vmin,vmax=vmax,extent=fitExtentR)
			axFit.set_title('Fit')
			axFit.set_yticks([]);axFit.set_xticks([])
			F.colorbar(caxFit,orientation='horizontal')
			axDiff=F.add_subplot(133)
			caxDiff=axDiff.imshow(Diff,cmap='cividis',extent=ExtentR)
			axDiff.set_title('Base - Fit')
			axDiff.set_yticks([]);axDiff.set_xticks([])
			F.colorbar(caxDiff,orientation='horizontal')

		# Subtract mean from Fit
		fitImgR+=meanDiff
		Diff=baseImgR-fitImgR # recompute difference
		meanDiff=np.nanmean(Diff) # recompute adjusted mean
		print('Removed constant offset. New mean diff: {}'.format(meanDiff))

		# Fit plane to difference
		if svdDnsmp=='auto':
			nTotal=fitMR*fitNR # total number of data points
			svdDnsmp=np.log(nTotal/5E4)/np.log(2)
		print('Downsample factor for SVD: {}'.format(svdDnsmp))
		DiffPlane=fitPlane(Diff,dtype='image',dx=baseXstepR,dy=baseYstepR,ds=svdDnsmp,includeAll=False,vocal=vocal,plot=False)

		# Remove difference from fit image
		print('Removing difference plane')

		# Remove plane at base resolution
		xcenter=0.; ycenter=0.  
		xMin=-np.floor(fitNR/2); xMax=np.ceil(fitNR/2) 
		yMin=-np.floor(fitMR/2); yMax=np.ceil(fitMR/2) 
		x=np.linspace(xMin,xMax,fitNR); y=np.linspace(yMin,yMax,fitMR) 
		Xplane,Yplane=np.meshgrid(x,y) # grid surface 
		DiffApprox=-1/DiffPlane.N[2]*(DiffPlane.N[0]*Xplane+DiffPlane.N[1]*Yplane)
		
		# Check ramp removal was successful at base resolution - Crt = "corrected"
		fitImgRCrt=fitImgR+DiffApprox # add back in the difference
		DiffRCrt=baseImgR-fitImgRCrt # corrected difference map
		DiffPlaneCrt=fitPlane(DiffRCrt,dtype='image',dx=baseXstepR,dy=baseYstepR,ds=svdDnsmp,includeAll=False)
		if vocal is True:
			print('Corrected difference fit: {}'.format(DiffPlaneCrt.N))

		meanDiffCrt=np.nanmean(DiffRCrt)
		meanAbsDiffCrt=np.nanmean(np.abs(DiffRCrt))
		print('--> Orig. mean difference: {}'.format(meanDiffOrig))
		print('--> Orig. mean abs difference: {}'.format(meanAbsDiffOrig))
		print('--> Final mean difference: {}'.format(meanDiffCrt))
		print('--> Final mean abs difference: {}'.format(meanAbsDiffCrt))

		# Construct approximate "difference plane" at original image resolution
		xcenter=0.; ycenter=0.  
		xMin=-np.floor(fitN/2); xMax=np.ceil(fitN/2) 
		yMin=-np.floor(fitM/2); yMax=np.ceil(fitM/2) 
		x=np.linspace(xMin,xMax,fitN)*fitXstep; y=np.linspace(yMin,yMax,fitM)*fitYstep 
		Xplane,Yplane=np.meshgrid(x,y) # grid surface 
		# Z = -1/C(AX+BY)
		DiffApprox=-1/DiffPlane.N[2]*(DiffPlane.N[0]*Xplane+DiffPlane.N[1]*Yplane)

		fitImgCrt=fitImg+DiffApprox # add back in the difference

		if plot is True:
			vmin=np.nanmin(fitImgCrt); vmax=np.nanmax(fitImgCrt)

			F=plt.figure()
			axDiff=F.add_subplot(121)
			caxDiff=axDiff.imshow(DiffApprox,cmap='viridis',
				vmin=vmin,vmax=vmax,extent=fitExtent)
			axDiff.set_title('Difference est.')
			F.colorbar(caxDiff,orientation='horizontal')
			axFit=F.add_subplot(122)
			caxFit=axFit.imshow(fitImgCrt,cmap='viridis',
				vmin=vmin,vmax=vmax,extent=fitExtent)
			axFit.set_title('Corrected fit')
			axFit.set_yticks([]); axFit.set_xticks([])
			F.colorbar(caxFit,orientation='horizontal')

		# Save fit image
		if saveImg is True:
			outName='{}_corrected.tif'.format(fitName)
			print('... Saving to: {}'.format(outName))
			driver=gdal.GetDriverByName('GTiff') 
			CorrectedImage=driver.Create(outName,FitDS.RasterXSize,FitDS.RasterYSize,1,gdal.GDT_Float32) 
			CorrectedImage.GetRasterBand(1).WriteArray(fitImgCrt) 
			CorrectedImage.SetProjection(FitDS.GetProjection()) 
			CorrectedImage.SetGeoTransform(FitDS.GetGeoTransform()) 
			CorrectedImage.FlushCache() 
			print('Saved!')
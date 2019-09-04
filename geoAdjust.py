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

		# Number of data points
		self.N=len(x)

		# Record coordinates
		self.x=x
		self.y=y
		self.site=site

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
	def plot(self,value=None,sitenames=False,title=None):
		F=plt.figure()
		ax=F.add_subplot(111)
		ax.scatter(self.x,self.y,color='k')
		if value is not None:
			cax=ax.scatter(self.x,self.y,c=value)
			F.colorbar(cax,orientation='vertical')
		if sitenames is True:
			for i in range(self.N):
				ax.text(self.x[i],self.y[i],self.site[i])
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
	def __init__(self,ImgObj,GPSobj,imgBand=1,gpsComponent='E',xWidth=3,yWidth=3,expc='median',vocal=False,plot=False,basename=None):
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
		if plot is True:
			Comparison.plot(title=gpsComponent,basename='{}_Orig'.format(basename))

		# Map properties
		nMap=ImgObj.RasterXSize 
		mMap=ImgObj.RasterYSize 

		T=ImgObj.GetGeoTransform()
		left=T[0]; xstep=T[1]; right=left+nMap*xstep 
		top=T[3]; ystep=T[5]; bottom=top+mMap*ystep
		xCenterMap=np.mean([left,right])
		yCenterMap=np.mean([bottom,top])
		extent=(left, right, bottom, top)

		# GPS properties
		nGPS=Comparison.Ngps
		xCenterGPS=np.mean(Comparison.easting)
		yCenterGPS=np.mean(Comparison.northing)

		# Report
		print('Fitting plane to differences')
		print('\tMap dimensions (px): {}x{}'.format(mMap,nMap))
		print('\tMap extent:\n\t\tleft: {}\tright {}'.format(left,right))
		print('\t\tbottom: {}\ttop: {}'.format(bottom,top))
		print('\t\txstep: {}\tystep: {}'.format(xstep,ystep))
		print('\t\txcenter: {}\tycenter: {}'.format(xCenterMap,yCenterMap))

		print('\tGPS extent:\n\t\txcenter: {}\tycenter: {}'.format(xCenterGPS,yCenterGPS))

		if plot is True:
			ax=plt.gca()
			ax.plot(xCenterMap,yCenterMap,'k+') # plot map center
			ax.plot(xCenterGPS,yCenterGPS,'ko') # plot GPS center

		# Difference stats
		meanDiff=np.nanmean(Comparison.differences)
		meanDiffOrig=np.nanmean(Comparison.differences)
		meanAbsDiffOrig=np.nanmean(np.abs(Comparison.differences))
		print('Uncorrected mean difference: {}'.format(meanDiff))

		## Fit plane to differences
		# Center data
		xGPS=Comparison.easting-xCenterGPS
		yGPS=Comparison.northing-yCenterGPS

		# Design matrix - [x,y,z]
		G=np.zeros((nGPS,3))
		G[:,0]=xGPS # x
		G[:,1]=yGPS # y
		G[:,2]=np.ones(nGPS) # c

		# Invert
		beta=np.linalg.inv(G.T.dot(G)).dot(G.T).dot(Comparison.differences)
		a=beta[0];b=beta[1];c=beta[2]
		print('\tSolution:\n\t\tA: {}\n\t\tB: {}\n\t\tC: {}'.format(a,b,c))

		# Construct plane at image dimensions/resolution
		xPlane=np.linspace(left-xCenterGPS+xstep/2,right-xCenterGPS-xstep/2,nMap)
		yPlane=np.linspace(bottom-yCenterGPS-ystep/2,top-yCenterGPS+ystep/2,mMap)
		Xplane,Yplane=np.meshgrid(xPlane,yPlane)
		Yplane=np.flipud(Yplane)
		DiffPlane=a*Xplane+b*Yplane+c 

		# Solve for hypothetical points and residuals
		hypDiffs=G.dot(beta) # hypothetical points on best fit plane
		residDiffs=Comparison.differences-hypDiffs # residual differences
		meanResidual=np.mean(residDiffs)
		meanExpResidual=np.mean(np.abs(residDiffs))
		print('\tMean of residuals: {}'.format(meanResidual))
		print('\tMean expected residual: {}'.format(meanExpResidual))

		if plot is True:
			from mpl_toolkits import mplot3d
			F=plt.figure()
			ax=F.add_subplot(111,projection='3d')
			ax.plot_surface(Xplane,Yplane,DiffPlane,color='k',alpha=0.25,zorder=1)
			for i in range(nGPS):
				ax.plot([xGPS[i],xGPS[i]],[yGPS[i],yGPS[i]],[Comparison.differences[i],hypDiffs[i]],'b')
			ax.plot(xGPS,yGPS,Comparison.differences,'bo',zorder=3)
			ax.set_title('Least squares fit to residuals')
			ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('diff')

		## Correct ramp + offset
		# Remove difference from original image
		img=ImgObj.GetRasterBand(imgBand).ReadAsArray() # load map from object
		CorrectedImage=img+DiffPlane # add difference back in
		print('Removed difference from plane')

		# Plot final results
		if plot is True:
			F=plt.figure()
			ax=F.add_subplot(131)
			cax=ax.imshow(img,extent=extent)
			ax.set_aspect(1); ax.set_title('Original')
			F.colorbar(cax,orientation='horizontal')
			ax=F.add_subplot(132)
			cax=ax.imshow(DiffPlane,extent=extent)
			ax.set_aspect(1); ax.set_title('Difference')
			F.colorbar(cax,orientation='horizontal')
			ax=F.add_subplot(133)
			cax=ax.imshow(CorrectedImage,extent=extent)
			ax.set_aspect(1); ax.set_title('Corrected')
			F.colorbar(cax,orientation='horizontal')
			if basename:
				outName='{}_FitResults.png'.format(basename)
				F.savefig(outName,dpi=300)

		## Save corrected image
		if basename:
			outName='{}_GPScorrected.tif'.format(basename)
			print('... Saving to: \n\t{}'.format(outName))
			driver=gdal.GetDriverByName('GTiff') 
			CorrectedImageDS=driver.Create(outName,ImgObj.RasterXSize,ImgObj.RasterYSize,1,gdal.GDT_Float32) 
			CorrectedImageDS.GetRasterBand(1).WriteArray(CorrectedImage)
			CorrectedImageDS.SetProjection(ImgObj.GetProjection()) 
			CorrectedImageDS.SetGeoTransform(ImgObj.GetGeoTransform()) 
			CorrectedImageDS.FlushCache() 
			print('Saved!')

			## Calculate and resport corrected stats
			ComparisonCrt=gpsCompare(CorrectedImageDS,GPSobj,imgBand=imgBand,gpsComponent=gpsComponent,
				xWidth=xWidth,yWidth=yWidth,expc=expc)
			if plot is True:
				ComparisonCrt.plot(title='{}_GPScorrected'.format(gpsComponent),
					basename='{}_GPScorrected'.format(basename))
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
	Base image must cover full spatial extent of fit image
	Base image should be coarser or equal resolution to fit image
	upFactor resamples the base image at finer resolution
	dwnFactor resamples the fit image at coarser resolution
	'''
	def __init__(self,BaseDS,FitDS,upFactor=None,plot=False,basename=None):
		print('Fitting image to base')

		# Upsample base image
		baseT=BaseDS.GetGeoTransform()
		if upFactor:
			newXres=baseT[1]/upFactor
			newYres=baseT[5]/upFactor
			BaseDS=gdal.Warp('Base_resampled.VRT',BaseDS,options=gdal.WarpOptions(
				format='VRT',xRes=newXres,yRes=newYres,
				resampleAlg='lanczos'))
			print('Upsampled base image by {}: New pixel ({}x{})'.format(upFactor,newXres,newYres))
			baseT=BaseDS.GetGeoTransform() # recompute transform

		## Find common geographic extent
		# Geographic extent
		baseN=BaseDS.RasterXSize; baseM=BaseDS.RasterYSize
		baseXstep=baseT[1]; baseYstep=baseT[5]
		baseLeft=baseT[0]
		baseRight=baseLeft+baseXstep*baseN
		baseTop=baseT[3]
		baseBottom=baseTop+baseYstep*baseM
		baseExtent=(baseLeft,baseRight,baseBottom,baseTop)

		fitT=FitDS.GetGeoTransform()
		fitN=FitDS.RasterXSize; fitM=FitDS.RasterYSize
		fitXstep=fitT[1]; fitYstep=fitT[5]
		fitLeft=fitT[0]
		fitRight=fitLeft+fitXstep*fitN 
		fitTop=fitT[3]
		fitBottom=fitTop+fitYstep*fitM
		fitExtent=(fitLeft,fitRight,fitBottom,fitTop)

		print('\tBase image: ({}x{}) {}'.format(baseM,baseN,baseT))
		print('\tFit image: ({}x{}) {}'.format(fitM,fitN,fitT))

		# Resample fit image to base image resolution
		FitDSresamp=gdal.Warp('Fit_resampled.VRT',FitDS,options=gdal.WarpOptions(
			format='VRT',xRes=baseXstep,yRes=baseYstep,
			resampleAlg='lanczos'))

		fitTresamp=FitDSresamp.GetGeoTransform() # recompute transform
		fitNresamp=FitDSresamp.RasterXSize; fitMresamp=FitDSresamp.RasterYSize
		fitXstepResamp=fitTresamp[1]; fitYstepResamp=fitTresamp[5]
		fitLeftResamp=fitTresamp[0]
		fitRightResamp=fitLeftResamp+fitXstepResamp*fitNresamp
		fitTopResamp=fitTresamp[3]
		fitBottomResamp=fitTopResamp+fitYstepResamp*fitMresamp
		fitExtentResamp=(fitLeftResamp,fitRightResamp,fitBottomResamp,fitTopResamp)
		print('Resampled Fit to xRes: {}; yRes: {}\n\t{}'.format(fitXstepResamp,fitYstepResamp,fitTresamp))

		# Crop base image to fit boundaries
		bounds=(fitLeftResamp,fitBottomResamp,fitRightResamp,fitTopResamp)
		BaseDSresamp=gdal.Warp('Base_resampled_cropped.VRT',BaseDS,options=gdal.WarpOptions(
			format='VRT',outputBounds=bounds))

		baseTresamp=BaseDSresamp.GetGeoTransform()
		baseNresamp=BaseDSresamp.RasterXSize; baseMresamp=BaseDSresamp.RasterYSize
		baseXstepResamp=baseTresamp[1]; baseYstepResamp=baseTresamp[5]
		baseLeftResamp=baseTresamp[0]
		baseRightResamp=baseLeftResamp+baseXstepResamp*baseNresamp
		baseTopResamp=baseTresamp[3]
		baseBottomResamp=baseTopResamp+baseYstepResamp*baseMresamp
		baseExtentResamp=(baseLeftResamp,baseRightResamp,baseBottomResamp,baseTopResamp)
		print('Cropped Base to (xmin ymin xmax ymax): {}\n\t{}'.format(bounds,baseTresamp))

		# Ensure coregistration
		assert baseExtentResamp==fitExtentResamp, '--- Images not properly aligned'
		print('Images aligned.')
		ExtentResamp=baseExtentResamp

		## Difference images
		# Difference Base - Fit
		fitImgResamp=FitDSresamp.GetRasterBand(1).ReadAsArray() # load as numpy arrays
		baseImgResamp=BaseDSresamp.GetRasterBand(1).ReadAsArray()
		DiffMap=baseImgResamp-fitImgResamp # take difference

		# Basic statistics
		meanDiffOrig=np.nanmean(DiffMap)
		meanAbsDiffOrig=np.nanmean(np.abs(DiffMap))
		print('Uncorrected mean difference: {}'.format(meanDiffOrig))

		if plot is True:
			vmin=np.nanmin(fitImgResamp); vmax=np.nanmax(fitImgResamp)

			F=plt.figure()
			axBase=F.add_subplot(131)
			caxBase=axBase.imshow(baseImgResamp,cmap='viridis',
				vmin=vmin,vmax=vmax,extent=baseExtentResamp)
			axBase.set_title('Base res')
			F.colorbar(caxBase,orientation='horizontal')
			axFit=F.add_subplot(132)
			caxFit=axFit.imshow(fitImgResamp,cmap='viridis',
				vmin=vmin,vmax=vmax,extent=fitExtentResamp)
			axFit.set_title('Fit res')
			axFit.set_yticks([]);axFit.set_xticks([])
			F.colorbar(caxFit,orientation='horizontal')
			axDiff=F.add_subplot(133)
			caxDiff=axDiff.imshow(DiffMap,cmap='viridis',extent=ExtentResamp)
			axDiff.set_title('Base - Fit res')
			axDiff.set_yticks([]);axDiff.set_xticks([])
			F.colorbar(caxDiff,orientation='horizontal')
			if basename:
				outName='{}_difference.png'.format(basename)
				F.savefig(outName,dpi=300)

		## Fit plane to difference map
		print('\tFitting plane to difference map')

		# Map coordinates
		xresamp=np.linspace(fitLeftResamp,fitRightResamp,fitNresamp)
		yresamp=np.linspace(fitBottomResamp,fitTopResamp,fitMresamp)
		Xresamp,Yresamp=np.meshgrid(xresamp,yresamp)
		Yresamp=np.flipud(Yresamp)

		# Design matrix - [x,y,c]
		nTotal=fitMresamp*fitNresamp # total number of data points
		G=np.zeros((nTotal,3)) # empty matrix
		G[:,0]=Xresamp.reshape(nTotal,1).squeeze(1)
		G[:,1]=Yresamp.reshape(nTotal,1).squeeze(1)
		G[:,2]=np.ones(nTotal)

		# Invert for fit parameters
		beta=np.linalg.inv(G.T.dot(G)).dot(G.T).dot(DiffMap.reshape(-1,1))
		a=beta[0];b=beta[1];c=beta[2]
		print('\tSolution:\n\t\tA: {}\n\t\tB: {}\n\t\tC: {}'.format(a,b,c))

		# Remove difference plane and calculate residuals
		DiffPlane=G.dot(beta).reshape(fitMresamp,fitNresamp)
		fitImgResampCrt=fitImgResamp+DiffPlane # corrected

		DiffCrt=baseImgResamp-fitImgResampCrt # recompute corrected differences
		meanDiffCrt=np.nanmean(DiffCrt)
		meanAbsDiffCrt=np.nanmean(np.abs(DiffCrt))

		if plot is True:
			vmin=np.nanmin(fitImgResamp); vmax=np.nanmax(fitImgResamp)

			F=plt.figure()
			ax=F.add_subplot(131)
			cax=ax.imshow(fitImgResamp,cmap='viridis',extent=ExtentResamp)
			ax.set_title('Uncrt fit')
			F.colorbar(cax,orientation='horizontal')
			ax=F.add_subplot(132)
			cax=ax.imshow(DiffPlane,cmap='viridis',extent=ExtentResamp)
			ax.set_title('Hyp diff')
			ax.set_yticks([]);ax.set_xticks([])
			F.colorbar(cax,orientation='horizontal')
			ax=F.add_subplot(133)
			cax=ax.imshow(fitImgResampCrt,cmap='viridis',extent=ExtentResamp)
			ax.set_title('Correct')
			ax.set_yticks([]);ax.set_xticks([])
			F.colorbar(cax,orientation='horizontal')
			if basename:
				outName='{}_corrected_differences.png'.format(basename)
				F.savefig(outName,dpi=300)

		## Apply corrections at original fit image dimensions and resolution
		print('Applying correction at full resolution:')
		print('\tDimensions: ({}x{})\t Resolution: ({}x{})'.format(fitM,fitN,fitXstep,fitYstep))

		# Construct hypothetical "difference plane" at original image resolution
		nTotal=fitM*fitN # total data points at full res
		x=np.linspace(fitLeft,fitRight,fitN)
		y=np.linspace(fitBottom,fitTop,fitM)
		X,Y=np.meshgrid(x,y)
		Y=np.flipud(Y)

		G=np.zeros((nTotal,3)) # full-res design matrix
		G[:,0]=X.reshape(nTotal,1).squeeze(1)
		G[:,1]=Y.reshape(nTotal,1).squeeze(1)
		G[:,2]=np.ones(nTotal)
		DiffPlane=G.dot(beta) # calculate hypo plane
		DiffPlane=DiffPlane.reshape(fitM,fitN)

		# Apply full-resolution correction
		fitImgOrig=FitDS.GetRasterBand(1).ReadAsArray()
		fitImgCrt=fitImgOrig+DiffPlane

		if plot is True:
			extent=(fitLeft, fitRight, fitBottom, fitTop)
			F=plt.figure()
			ax=F.add_subplot(121)
			ax.set_title('Full res Orig')
			cax=ax.imshow(fitImgOrig,cmap='viridis',extent=extent)
			ax.set_aspect(1)
			F.colorbar(cax,orientation='horizontal')
			ax=F.add_subplot(122)
			ax.set_title('Full res Correct')
			cax=ax.imshow(fitImgCrt,cmap='viridis',extent=extent)
			ax.set_aspect(1)
			F.colorbar(cax,orientation='horizontal')

		# Save fit image
		if basename:
			outName='{}_corrected.tif'.format(basename)
			print('... Saving to: {}'.format(outName))
			driver=gdal.GetDriverByName('GTiff') 
			CorrectedImage=driver.Create(outName,FitDS.RasterXSize,FitDS.RasterYSize,1,gdal.GDT_Float32) 
			CorrectedImage.GetRasterBand(1).WriteArray(fitImgCrt) 
			CorrectedImage.SetProjection(FitDS.GetProjection()) 
			CorrectedImage.SetGeoTransform(FitDS.GetGeoTransform()) 
			CorrectedImage.FlushCache() 
			print('Saved!')

		print('--> Orig. mean difference: {}'.format(meanDiffOrig))
		print('--> Orig. mean abs difference: {}'.format(meanAbsDiffOrig))
		print('--> Final mean difference: {}'.format(meanDiffCrt))
		print('--> Final mean abs difference: {}'.format(meanAbsDiffCrt))



#############################
### --- Miscellaneous --- ###
#############################
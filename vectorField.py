import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal

class vField:
	'''
	EW is the east-west or x field
	NS is the north-south or y field
	xWin is the window size (px) in x - must be odd
	yWin is the window size (px) in y - must be odd
	xstep is the step size (px) in the x direction - must be even
	ystep is the step size (px) in the y direction - must be even
	'''

	# --- Create vectors ---
	def __init__(self,EW,NS,xWin,yWin,xstep,ystep,geoTransform=None,vocal=False):
		assert EW.shape==NS.shape, print('Images must be same size')	

		# Image/spatial parameters
		nImg=NS.shape[1] # image dimension in x
		mImg=EW.shape[0] # image dimension in y

		# Parse geo transform if specified
		if geoTransform is not None:
			left=geoTransform[0] # upper left x-coordinate
			top=geoTransform[3] # upper left y-coordinate
			dx=geoTransform[1] # pixel size in x
			dy=geoTransform[5] # pixel size in y
			right=left+nImg*dx # easternmost extent
			bottom=top+mImg*dy # southernmost extent
			self.extent=(left, right, bottom, top)
		else:
			ULx=0; ULy=0
			dx=1; dy=1
			self.extent=(0, nImg, 0, mImg)


		# Estimate number of steps based on pixel size
		xWinWidth=int(xWin/2) # half-window width in x
		yWinWidth=int(yWin/2) # half-window width in y
		ystep=np.abs(ystep); xstep=np.abs(xstep)

		n_y_steps=int(mImg/ystep) # number of steps in y
		m_x_steps=int(nImg/xstep) # number of steps in x
		self.total_points=n_y_steps*m_x_steps # total number of measurements

		y2step=int(ystep/2) # half step in y
		x2step=int(xstep/2) # half step in x

		if vocal is True:
			print('\tSteps in x: %i' % m_x_steps)
			print('\tSteps in y: %i' % n_y_steps)

		xlocations=np.arange(x2step,nImg,xstep)
		ylocations=np.arange(y2step,mImg,ystep)

		# Determine vectors
		self.x=np.zeros((self.total_points))
		self.y=np.zeros((self.total_points))
		self.u=np.zeros((self.total_points))
		self.v=np.zeros((self.total_points))

		k=0
		for i in range(n_y_steps):
			for j in range(m_x_steps):
				self.x[k]=xlocations[j]
				self.y[k]=ylocations[i]

				x_region=np.arange(xlocations[j]-xWinWidth,xlocations[j]+xWinWidth+1).astype(int)
				y_region=np.arange(ylocations[i]-yWinWidth,ylocations[i]+yWinWidth+1).astype(int)
				
				self.u[k]=np.nanmean(EW[y_region,x_region])
				self.v[k]=np.nanmean(NS[y_region,x_region])

				k+=1

		self.x=self.x*dx+left
		self.y=self.y*dy+top


	# --- Plot on image of choice ---
	def plotVectors(self,img,arrow_color='k',cmap='viridis',headwidth=3,headlength=5):
		F=plt.figure()
		ax=F.add_subplot(111)
		cax=ax.imshow(img,cmap=cmap,extent=self.extent,zorder=1)
		ax.quiver(self.x,self.y,self.u,self.v,
			units='xy',color=arrow_color,
			headwidth=headwidth,headlength=headlength,
			zorder=2)
		F.colorbar(cax,orientation='vertical')
		ax.set_aspect(1)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Image processing algorithms 
# by Rob Zinke, 2019 
# 
# In all cases: 
#	I is the reference image 
#	ds is the downsample factor (power of 2) 
#	w is the kernel width 
#	k is the intensity factor 
#	d is direction 
#	ktype is kernel type 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle 
from matplotlib.patches import Circle 
from mpl_toolkits import mplot3d 
from scipy import signal as sig 
from scipy import interpolate as intp 
from scipy import integrate 
from fitPlane import *

# --- Formulae for later --- 
def gauss(x,mu,sig):
	g=1/(sig*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu)/sig)**2)
	return g

def erf(x,mu,sig):
	g=gauss(x,mu,sig) # create Gaussian 
	G=integrate.cumtrapz(g,x,initial=0) 
	return G 

# --- Quick plot --- 
# Plot a single image for later 
def quickPlot(I,cmap='Greys_r',title=None,colorbar=False): 
	F=plt.figure() # spawn new figure 
	ax=F.add_subplot(111) # add axis 
	cax=ax.imshow(I,cmap=cmap) # plot image 
	ax.set_xticks([]); ax.set_yticks([]) # remove tick marks 
	if title is not None: 
		ax.set_title(title) 
	if colorbar is not False: 
		if colorbar in ['horizontal']: 
			orientation='horizontal' 
		else: 
			orientation='vertical' 
		F.colobar(cax,orientation=orientation)

def kmeans(data,k,max_iterations=20,centroids='auto',vocal=False,plot=False):
	# INPUTS
	#	data are the data for which the clusters are to be analyzed
	#	 each column is a dimension in space
	#	k is the number of clusters
	#	max_iterations is the maximum number of iterations, otherwise
	#	 the routine will cease when convergence is reached
	#	centroids can either be pre-defined or chosen at random
	# OUTPUTS
	#	Centroids is the matrix of means

	# Setup
	dataMin=np.min(data,axis=0)
	dataMax=np.max(data,axis=0)
	dataShape=data.shape
	np.random.seed(0)
	Dists=np.zeros((dataShape[0],k))
	# Pick initial centroids
	if centroids in 'auto':
		Centroids=np.linspace(dataMin,dataMax,k)
	Centroids_new=np.zeros((k,dataShape[1]))
	if vocal is True:
		print('Initial centroids:\n',Centroids)
	# Find closest centroids
	def find_closest_centroid(data,k):
		for i in range(k):
			# Subract each centroid from the data
			#  and compute Euclidean distance
			Dists[:,i]=np.linalg.norm(data-Centroids[i,:],
				ord=dataShape[1],axis=1)
		return Dists
	# Loop through iterations
	while(max_iterations): # for each iteration...
	# 1. Calculate distance to points
		Dists=find_closest_centroid(data,k)
		Centroid_ndx=np.argmin(Dists,axis=1)
		if vocal is True:
			print('Distances\n',np.hstack([Dists,Centroid_ndx.reshape(-1,1)]))
	# 2. Assign data points to centroids
		for j in range(k):
			cluster_mean=np.mean(data[Centroid_ndx==j],axis=0)
			if vocal is True:
				print('cluster means\n',cluster_mean)
			Centroids_new[j,:]=cluster_mean
		if not np.sum(Centroids_new-Centroids):
			break
		Centroids=Centroids_new
		max_iterations-=1
	# Plot
	Dists=find_closest_centroid(data,k) # final calculation
	Centroid_ndx=np.argmin(Dists,axis=1) # final indices
	if plot is True:
		F=plt.figure()
		ax1=F.add_subplot(1,1,1)
		for i in range(k):
			plot_data=data[Centroid_ndx==i][::100]
			ax1.plot(plot_data,np.zeros(plot_data.shape),'o')
		ax1.plot(Centroids[:,0],np.zeros(len(Centroids[:,0])),'k*')
	return Centroids


################################
### --- Standard Filters --- ###
################################

# --- Delta --- 
def delta(I,ds=0):
	ds=int(2**ds); I=I[::ds,::ds] 
	h=np.array([[0,0,0],[0,1,0],[0,0,0]])
	I=sig.convolve2d(I,h,'same')
	return I

# --- Edge detection --- 
def edgeDetect(I,k=1,ds=0): 
	ds=int(2**ds); I=I[::ds,::ds] 
	h=np.array([[-k/8,-k/8,-k/8],[-k/8,k,-k/8],[-k/8,-k/8,-k/8]])
	I=sig.convolve2d(I,h,'same') 
	I=I-I.min(); I=255*I/I.max() 
	return I 

# --- Moving mean window --- 
def meanWin(I,w=2,ds=0): 
	ds=int(2**ds); I=I[::ds,::ds] 
	h=np.ones((w,w))/(w**2) 
	I=sig.convolve2d(I,h,'same') 
	return I  

# --- Generic blur --- 
def blur(I,ds=0):
	ds=int(2**ds); I=I[::ds,::ds] 
	h=np.array([[1/8,1/4,1/8],[1/4,1,1/4],[1/8,1/4,1/8]])
	I=sig.convolve2d(I,h,'same') 
	I=I-I.min(); I=255*I/I.max() 
	return I 

# --- Gaussian blur --- 
def gaussBlur(I,w=3,ds=0):
	ds=int(2**ds); I=I[::ds,::ds] 
	# n is an odd integer 
	if w%2==0: 
		w=int(w-1); print('Warning: Rounded w to odd!') 
	w2=np.floor(w/2) # half-width 
	# Build Gaussian 
	x=np.linspace(-w2,w2,w) # x-array 
	y=gauss(x,0,w2/3) # 3 std deviations 
	h=y.reshape(-1,1)*y.reshape(1,-1) 
	# Run filter 
	I=sig.convolve2d(I,h,'same') 
	I=I-I.min(); I=255*I/I.max() 
	return I 

# --- Shading --- 
def shade(I,d='NW',ds=0):
	ds=int(2**ds); I=I[::ds,::ds] 
	if d=='N':
		h=np.array([[0,0,0],[0,1,0],[0,-1,0]])
	elif d=='NE':
		h=np.array([[0,0,0],[0,1,0],[-1,0,0]])
	elif d=='E':
		h=np.array([[0,0,0],[-1,1,0],[0,0,0]])
	elif d=='SE':
		h=np.array([[-1,0,0],[0,1,0],[0,0,0]])
	elif d=='S':
		h=np.array([[0,-1,0],[0,1,0],[0,0,0]])
	elif d=='SW':
		h=np.array([[0,0,-1],[0,1,0],[0,0,0]])
	elif d=='W':
		h=np.array([[0,0,0],[0,1,-1],[0,0,0]])
	elif d=='NW':
		h=np.array([[0,0,0],[0,1,0],[0,0,-1]])
	I=sig.convolve2d(I,h,'same') 
	I=I-I.min(); I=255*I/I.max() 
	return I 

# --- Sharpen --- 
def sharpen(I,k=2,w=3,ds=0): 
	ds=int(2**ds); I=I[::ds,::ds] 
	Ik=k*I # multiply by k 
	Im=meanWin(I,w=w) # divide by mean 
	I=Ik-Im # difference 
	return I 


#################################
### --- Frequency Filters --- ###
################################# 

# --- Show Fourier Spectrum --- 
def spectrum(I,dx=1,coords='polar',vocal=False): 
	# Image shape 
	m,n=I.shape 
	# Fourier transform 
	IF=np.fft.fft2(I) 
	IF=np.fft.fftshift(IF) # shift zero-freq to center 
	# Output units 
	fs=1/dx # sampling frequency = 1/unit spacing 
	fN=0.5*fs # Nyquist freq = 1/2 sampling freq 
	xMin=-n/2; xMax=n/2 # <--- convert these to actual units 
	yMin=-m/2; yMax=m/2 
	extent=(-fN,fN,-fN,fN) 
	if vocal is True: 
		print('Image shape: ',I.shape)
		print('Sampling freq: %f' % (fs)) 
		print('Nyquist freq:  %f' % (fN)) 
	# Convert representation 
	if coords is 'rectangular': 
		IFa=IF.real # real component 
		LabelA='real' 
		IFb=IF.imag # imaginary component 
		LabelB='imaginary' 
	if coords is 'polar': 
		IFa=np.abs(IF) # magnitude 
		IFa=np.log10(IFa) # log scale for clarity 
		LabelA='Log magnitude' 
		IFb=np.angle(IF) # phase 
		LabelB='Phase' 
	# Plot 
	F=plt.figure() 
	ax1=F.add_subplot(1,2,1) 
	cax1=ax1.imshow(IFa,extent=extent) 
	ax1.set_title(LabelA) 
	F.colorbar(cax1,orientation='horizontal') 
	ax2=F.add_subplot(1,2,2) 
	cax2=ax2.imshow(IFb,extent=extent) 
	ax2.set_title(LabelB) 
	F.colorbar(cax2,orientation='horizontal') 

# --- Basic frequency filter --- 
def freqFilt(I,fcut,dx=1,taper=None,ftype='low',vocal=False,plot=False): 
	m,n=I.shape 
	m2=m/2; n2=n/2 # half-widths 
	# Sample rates 
	fs=1/dx   # sampling frequency 
	fN=0.5*fs # Nyquist frequency 
	# Fourier domain 
	IF=np.fft.fft2(I) 
	IF=np.fft.fftshift(IF) 
	IFcut=IF.copy() 
	if taper is None: 
		xcut1=int(n2-n2*fcut) 
		xcut2=int(n2+n2*fcut) 
		ycut1=int(m2-m2*fcut) 
		ycut2=int(m2+m2*fcut) 
		K=np.zeros((m,n)) # empty filter kernel 
		K[ycut1:ycut2,xcut1:xcut2]=1. 
	elif taper is 'gauss': 
		x=gauss(np.linspace(-n2,n2,n),0,n2*fcut) 
		x=x/x.max() 
		y=gauss(np.linspace(-m2,m2,m),0,m2*fcut) 
		y=y/y.max() 
		K=np.dot(y.reshape(-1,1),x.reshape(1,-1)) 
		K=K/K.max() 
	if ftype.lower()=='hi' or ftype.lower()=='high': 
		K=1-K 
	IFcut=K*IFcut 
	# Outputs 
	if vocal is True: 
		print('Image shape: %i x %i' % (m,n)) 
		print('Frequency filter') 
		print('\tMode: %s' % (ftype))
		print('\tsample width: %f' % dx) 
		print('\tsample  freq: %f' % fs) 
		print('\tNyquist freq: %f' % fN) 
		print('\tcutoff  freq: %f' % fcut)
	if plot is True: 
		# Mask array 
		IFcut[IFcut==0.]=1E-9 # no zeros for plotting 
		IFcut=np.ma.array(IFcut,mask=(K==0)) 
		# Plot 
		F=plt.figure() 
		ax1=F.add_subplot(1,3,1) 
		cax1=ax1.imshow(np.log10(np.abs(IF))) 
		F.colorbar(cax1,orientation='horizontal')
		ax2=F.add_subplot(1,3,2) 
		cax2=ax2.imshow(K) 
		F.colorbar(cax2,orientation='horizontal')
		ax3=F.add_subplot(1,3,3) 
		cax3=ax3.imshow(np.log10(np.abs(IFcut))) 
		F.colorbar(cax3,orientation='horizontal')
	# Reconstruct image 
	IFcut=np.fft.ifftshift(IFcut) 
	I=np.fft.ifft2(IFcut) 
	I=I.real 
	return I 


###############################
### --- Slope/gradients --- ###
###############################

# --- X-derivative ---
def dzdx(I,dx=1.,ktype='sobel'):
	# dx is x cell size
	if ktype=='roberts':
		h=np.array([[0.,1.],[-1.,0.]])
		h=h/(2*dx)
	elif ktype=='prewitt':
		h=np.array([[1.,0.,-1.],[1.,0.,-1.],[1.,0.,-1.]])
		h=h/(6*dx)
	elif ktype=='sobel':
		h=np.array([[1.,0.,-1.],[2.,0.,-2.],[1.,0.,-1.]])
		h=h/(8*dx) # divide by cell spacing in x
	elif ktype=='scharr':
		h=np.array([[3.,0.,-3.],[10,0.,-10],[3.,0.,-3.]])
		h=h/(32*dx)
	else: 
		print('Kernel types: roberts,prewitt,sobel,scharr')
	dX=sig.convolve2d(I,h,'same')
	return dX

# --- Y-derivative ---
def dzdy(I,dy=1.,ktype='sobel'):
	# dy is y cell size
	if ktype=='roberts':
		h=np.array([[1.,0.],[0.,-1.]])
		h=h/(2*dy)
	elif ktype=='prewitt':
		h=np.array([[1.,1.,1.],[0.,0.,0.],[-1.,-1.,-1.]])
		h=h/(6*dy)
	elif ktype=='sobel':
		h=np.array([[1.,2.,1.],[0.,0.,0.],[-1.,-2.,-1.]])
		h=h/(8*dy) # divide by cell spacing in y
	elif ktype=='scharr':
		h=np.array([[3.,10,3.],[0.,0.,0.],[-3.,-10,-3.]])
		h=h/(32*dy)
	else: 
		print('Kernel types: roberts,prewitt,sobel,scharr')
	dY=sig.convolve2d(I,h,'same')
	return dY

# --- Slope and aspect ---
def slope(I,dx=1.,dy=1.,ktype='sobel'):
	# dx is x cell size
	# dy is y cell size
	dX=dzdx(I,dx=dx,ktype=ktype)
	dY=dzdy(I,dy=dy,ktype=ktype)
	dZ=np.sqrt(dX**2+dY**2)
	slope=np.rad2deg(np.arctan(dZ))
	aspect=np.rad2deg(np.arctan2(dX,dY)) 
	aspect[aspect>0]=360-aspect[aspect>0] 
	aspect[aspect<0]=-aspect[aspect<0] 
	return slope, aspect 

# --- Gradient --- 
class grad: 
	# dx is the cell size in x 
	# dy is the cell size in y 
	def __init__(self,I,dx,dy,ktype='scharr'): 
		# Establish kernel 
		ktype=ktype.lower(); self.ktype=ktype 
		if ktype=='roberts': 
			h=np.array([[0+1.j,1+0.j],[-1+0.j,0-1.j]]) 
			h.real=h.real/(2*dx); h.imag=h.imag/(2*dy) 
		elif ktype=='prewitt': 
			h=np.array([[1+1.j,0+1.j,-1+1.j],
						[1+0.j,0+0.j,-1+0.j],
						[1-1.j,0-1.j,-1-1.j]]) 
			h.real=h.real/(6*dx); h.imag=h.imag/(6*dy) 
		elif ktype=='sobel': 
			h=np.array([[1+1.j,0+2.j,-1+1.j],
						[2+0.j,0+0.j,-2+0.j],
						[1-1.j,0-2.j,-1-1.j]]) 
			h.real=h.real/(8*dx); h.imag=h.imag/(8*dy) 
		elif ktype=='scharr': 
			h=np.array([[3+3.j,0+10.j,-3+3.j],
						[10+0.j,0+0.j,-10+0.j],
						[3-3.j,0-10.j,-3-3.j]])
		# Calculate gradient map 
		G=sig.convolve(I,h,'same') 
		self.dzdx=G.real 
		self.dzdy=G.imag 
		self.grad=np.abs(G) 
		self.az=np.angle(G) 


################################
### --- Spatial analysis --- ###
################################

# --- Shift by integers --- 
def imgShift(I,xshift,yshift,outVal=None,crop=False): 
	xshift=int(xshift); yshift=int(yshift) 
	I=np.concatenate([I[yshift:,:],I[:yshift,:]],axis=0) 
	I=np.concatenate([I[:,xshift:],I[:,:xshift]],axis=1) 
	if outVal is not None: 
		pass 
	if crop is True: 
		pass 
	return I 

# --- Integer shift correlation --- 
def intCorr(I1,I2,window=None,vocal=False,plot=False): 
	# Setup 
	m,n=I1.shape # size of images 
	if vocal is True: 
		print('I1 size:',I1.shape) 
		print('I2 size:',I2.shape) 
		print('Window:',window)
	assert I1.shape == I2.shape 
	I1=I1.copy(); I2=I2.copy() 
	# Window 
	W=np.ones((m,n)) 
	if window is 'hamming' or window is 'Hamming': 
		W*=np.hamming(n).reshape(1,-1) 
		W*=np.hamming(m).reshape(-1,1) 
	if window is 'blackman' or window is 'Blackman': 
		W*=np.blackman(n).reshape(1,-1) 
		W*=np.blackman(m).reshape(-1,1) 
	I1*=W; I2*=W # apply windows
	# Fourier transforms 
	IF1=np.fft.fft2(I1) 
	IF2=np.fft.fft2(I2) 
	C=IF1*IF2.conjugate() 
	C=np.fft.ifft2(C) # back to spatial domain 
	C=C.real # real component only 
	X,Y=np.meshgrid(np.arange(n),np.arange(m)) 
	xshift=X[C==C.max()].astype(int) # shift in x 
	yshift=Y[C==C.max()].astype(int) # shift in y 
	if vocal is True: 
		xxshift=n-xshift # alternative 
		yyshift=m-yshift # alternative 
		print('X shift: %i (-%i)' % (xshift,xxshift))
		print('Y shift: %i (-%i)' % (yshift,yyshift)) 
	if plot is True: 
		F=plt.figure() 
		ax1=F.add_subplot(1,1,1) 
		cax1=ax1.imshow(C,cmap='cividis') 
		F.colorbar(cax1,orientation='horizontal') 
	return xshift, yshift 


##################################
### --- Transforms Scaling --- ###
##################################

# --- Normalize --- 
# Normalize to values 0 - 255 
def imgNorm(I,Imin=0,Imax=255): 
	I=I-I.min()-Imin # shift to min value 
	I=Imax*I/I.max() # stretch to max value 
	return I 

# --- Convert to binary --- 
def binary(I,pct=50,value=None,low=0,high=1,ds=0,vocal=False): 
	ds=int(2**ds); I=I[::ds,::ds] 
	m,n=I.shape 
	if value is not None: 
		threshold=value # use strict value 
	else: 
		threshold=np.percentile(I,(pct)) # use percentage 
	if vocal is True: 
		print('Threshold: %.1f' % (threshold))
	I[I<=threshold]=low 
	I[I>threshold]=high 
	if vocal is True: 
		N0=np.sum(I==low) 
		N1=np.sum(I==high) 
	return I 

# --- Erosion dilation --- 
def erosionDilation(I,ErodeDilate,maskThreshold=0.5,
	binaryPct=None,binaryValue=None,ds=0,vocal=False):
	# INPUTS
	#	I is the image (preferably binary)
	#	ErodeDilate is a switch 'erode'/'dilate'
	#	binaryPct, binaryValue initiate conversion to binary
	# OUTPUTS
	#	binary mask
	Iout=I.copy()
	ErodeDilate=ErodeDilate.lower()
	# Convert to 0,1 binary if required 
	if binaryPct is not None and binaryValue is None: 
		Iout=binary(I,pct=binaryPct,ds=ds,vocal=vocal) 
	elif binaryValue is not None:
		Iout=binary(I,value=binaryValue,ds=ds,vocal=vocal)
	else:
		ds=int(2**ds); Iout=I[::ds,::ds]
	# Compute average window 
	k=np.ones((3,3))/9 
	C=sig.convolve2d(Iout,k,'same') 
	if ErodeDilate=='erode': 
		Iout[C<=maskThreshold]=0 
	elif ErodeDilate=='dilate': 
		Iout[C>=1-maskThreshold]=1 
	return Iout  

# --- Linear transform --- 
def linearTransform(I,B0,B1,ds=0,interp_kind='linear',show_gamma=False): 
	# Setup 
	ds=2**ds; I=I[::ds,::ds] 
	m=I.shape[0]; n=I.shape[1] 
	I=I.reshape(1,-1) 

	# Transform curve 
	x=np.arange(0,256) 
	G=B1*x+B0 # y = mx + b 

	F=intp.interp1d(x,G,kind=interp_kind) 
	Itrans=F(I) 
	Itrans[Itrans<0]=0 
	Itrans[Itrans>255]=255 

	# Plot histogram? 
	if show_gamma is not False: 
		# Determine number of bins 
		if type(show_gamma)==int: 
			nbins=show_gamma 
		else: 
			nbins=255 
		# Compute histograms 
		H1,H1edges=np.histogram(I,bins=nbins) 
		H1cntrs=H1edges[:-1]+np.diff(H1edges)/2 
		H2,H2edges=np.histogram(Itrans,bins=nbins) 
		H2cntrs=H2edges[:-1]+np.diff(H2edges)/2 
		# Format curves 
		H1cntrs=np.pad(H1cntrs,(1,1),'constant',
			constant_values=(H1edges[0],H1edges[-1])) 
		H1=np.pad(H1,(1,1),'constant'); H1=255*H1/H1.max() 
		H2cntrs=np.pad(H2cntrs,(1,1),'constant',
			constant_values=(H2edges[0],H2edges[-1])) 
		H2=np.pad(H2,(1,1),'constant'); H2=255*H2/H2.max() 
		# Relative resolving power 
		R=B1*H1 
		R[H1cntrs<=-B0/B1]=0.; R[H1cntrs>=(255-B0)/B1]=0. 
		# Plot curves 
		FigG=plt.figure('Linear Transform') 
		ax1=FigG.add_subplot(111) # Gamma curve 
		cax1=ax1.plot(x,G,'r',linewidth=2) 
		ax1.axis((0,255,0,255));ax1.set_aspect(1) 
		ax1.fill(H1cntrs,H1,color=(0.4,0.5,0.5),alpha=1,label='orig') 
		ax1.plot(H1cntrs,R,'g--',label='reslv') 
		ax1.fill(H2cntrs,H2,color=(0,0,1),alpha=0.5,label='trnsf') 
		ax1.legend()  
	# Output 
	Itrans=Itrans.reshape(m,n) 
	return Itrans 

# --- Gaussian transform --- 
def gaussTransform(I,A,B,ds=0,interp_kind='linear',show_gamma=False):
	# Setup
	ds=2**ds; I=I[::ds,::ds]
	m=I.shape[0]; n=I.shape[1]
	I=I.reshape(1,-1)
	# Transform curve 
	x=np.arange(0,256)
	G=255*erf(x,A,B) # Gaussian error function
	F=intp.interp1d(x,G,kind=interp_kind)
	Itrans=F(I)
	Itrans[Itrans<0]=0
	Itrans[Itrans>255]=255
	# Plot histogram?
	if show_gamma is not False:
		# Determine number of bins
		if type(show_gamma)==int:
			nbins=show_gamma
		else:
			nbins=255
		# Compute histograms 
		H1,H1edges=np.histogram(I,bins=int(256/4)) 
		H1cntrs=H1edges[:-1]+np.diff(H1edges)/2 
		H2,H2edges=np.histogram(Itrans,bins=int(256/4)) 
		H2cntrs=H2edges[:-1]+np.diff(H2edges)/2 
		# Format curves 
		H1cntrs=np.pad(H1cntrs,(1,1),'constant',
			constant_values=(H1edges[0],H1edges[-1])) 
		H1=np.pad(H1,(1,1),'constant'); H1=255*H1/H1.max() 
		H2cntrs=np.pad(H2cntrs,(1,1),'constant',
			constant_values=(H2edges[0],H2edges[-1])) 
		H2=np.pad(H2,(1,1),'constant'); H2=255*H2/H2.max() 
		# Relative resolving power 
		R=H1*gauss(H1cntrs,A,B); R=255*R/R.max() 

		# Plot curves 
		FigG=plt.figure('Gaussian Transform') 
		ax1=FigG.add_subplot(111) # Gamma curve 
		cax1=ax1.plot(x,G,'r',linewidth=2) 
		ax1.axis((0,255,0,255));ax1.set_aspect(1) 
		ax1.fill(H1cntrs,H1,color=(0.4,0.5,0.5),alpha=1,label='orig') 
		ax1.plot(H1cntrs,R,'g--',label='resvl')
		ax1.fill(H2cntrs,H2,color=(0,0,1.0),alpha=0.5,label='trnsf') 
		ax1.legend() 
	# Output 
	Itrans=Itrans.reshape(m,n) 
	return Itrans 

# --- Histogram equalize --- 
def equalize(I,nbins=256,ds=0,vocal=False,plot=False): 
	# Setup 
	ds=int(2**ds); I=I[::ds,::ds] 
	m,n=I.shape # orig shape 
	Imin=I.min(); Imax=I.max() 
	Ivals=I.copy().reshape(1,-1).squeeze(0) # single row 
	N=len(Ivals) # total number of pixels 
	# Build initial histogram 
	H0,edges=np.histogram(Ivals,bins=nbins) 
	cntrs=edges[:-1]+np.diff(edges)/2 
	# Calculate probability distributions 
	P0=H0/N # probability mass 
	C0=np.cumsum(P0) # cumulative prob 
	C0[0]=0 # start with zero 
	C0=(Imax-Imin)*C0+Imin # value range 
	if vocal is True: 
		print('Equalizing')
		print('\tImin: %f\tImax: %f' % (Imin,Imax)) 
		print('\tCmin: %f\tCmax: %f' % (C0.min(),C0.max()))
	# Inverse distribution function 
	Cintp=intp.interp1d(edges[:-1],C0,kind='linear',bounds_error=False,fill_value=Imax) 
	Ieq=Cintp(I) 
	# Plot 
	Heq,edges=np.histogram(Ieq,bins=nbins) 
	Peq=Heq/N 
	Ceq=(Imax-Imin)*np.cumsum(Peq)+Imin 
	X=np.hstack([cntrs.reshape(-1,1),np.ones((nbins,1))]) 
	fit=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Ceq.reshape(-1,1)) 
	if plot is not False: 
		F=plt.figure() 
		ax1=F.add_subplot(2,1,1) 
		ax1.bar(cntrs-0.4,P0,color='k') 
		ax1.bar(cntrs+0.4,Peq,color='b')
		ax1.set_ylabel('PDF') 
		ax2=F.add_subplot(2,1,2) 
		ax2.plot(cntrs,C0,'k',linewidth=2,zorder=1)
		ax2.plot(cntrs,X.dot(fit),color=(0.5,0.6,1),zorder=2) 
		ax2.plot(cntrs,Ceq,'b',linewidth=2,zorder=3)
		ax2.set_ylabel('CDF') 
	if vocal is True: 
		print('\tfit: %f' % (fit[0]))
	return Ieq 

# --- Intervals --- 
# Bin values into intervals 
def intervals(I,bins,mode='values',nbins=100,vocal=False,plot=False): 
	# INPUTS 
	#	I is the image to be analyzed 
	#	B is the bin centers
	#	 if B is an integer, B break values will be used, resulting in B+1 bins 
	#	 if B is a list, the values will be bin centers 
	#	 B values can represent either image values, or percentiles 
	#	  depending on user-specified 'mode' parameter 
	#	mode is the distribution type, can be either 'values' or 'perc' 
	# OUTPUT 
	#	I is the binned image 

	## Setup
	I_out=I.copy()
	# Statistics
	Min=I.min(); Max=I.max() # image statistics
	H,Hedges=np.histogram(I.reshape(-1,1),nbins) # histogram
	Hcntrs=Hedges[:-1]+np.diff(Hedges)/2 # hist bin centers
	C=np.cumsum(H)/np.sum(H) # cumulative function
	# Interpolation functions
	Hintrp=intp.interp1d(Hcntrs,H,kind='linear',
		bounds_error=False,fill_value=np.nan)
	Cintrp=intp.interp1d(Hcntrs,C,kind='linear',
		bounds_error=False,fill_value=np.nan)
	# Define bins and breaks
	if mode in ['values','vals']:
		# Evenly spaced bins
		if type(bins)==int:
			nBins=int(bins) # number of bins
			nBrks=nBins-1+2 # add 2 for ends of range
			brks=np.linspace(Min,Max,nBrks)
			bins=brks[:-1]+np.diff(brks)/2 # bins
			prctBins=Cintrp(bins) # bin percentiles
			prctBrks=Cintrp(brks) # break percentiles
			prctBrks[0]=C.min(); prctBrks[-1]=C.max() # ends
		# User-specified bin centers
		else:
			nBins=len(bins) # number of bins
			nBrks=nBins-1+2 # number of breaks
			if type(bins)==list:
				bins=np.array(bins) # use array
			brks=np.zeros(nBrks) # empty array for break values
			brks[1:-1]=bins[:-1]+np.diff(bins)/2 # breaks from bins
			brks[0]=Min; brks[-1]=Max # end values
			prctBins=Cintrp(bins) # bin percentiles
			prctBrks=Cintrp(brks) # break percentiles
			prctBrks[0]=C.min(); prctBrks[-1]=C.max() # ends
	elif mode in ['percentiles','perc','percent','pct']:
		# Evenly spaced percentiles
		if type(bins)==int:
			nBins=int(bins) # number of bins
			nBrks=nBins-1+2 # add 2 for ends of range
			prctBrks=np.linspace(C.min(),C.max(),nBrks) # breaks by pctile
			# Interpolate break values
			BrkInterp=intp.interp1d(C,Hcntrs,kind='linear',
				bounds_error=False,fill_value=np.nan) # interpolation
			brks=BrkInterp(prctBrks) # interpolate break values
			prctBins=prctBrks[:-1]+np.diff(prctBrks)/2
			bins=BrkInterp(prctBins) # interpolate bin values
		# User-specified percentiles
		else:
			nBins=len(bins) # number of bins
			nBrks=nBins-1+2 # number of breaks
			if type(bins)==list:
				bins=np.array(bins) # use array
			if np.max(bins)>1.0:
				bins=bins/100 # adjust percentiles to fractions
			prctBrks=np.zeros(nBrks) # empty array for break prcntiles
			prctBrks[1:-1]=bins[:-1]+np.diff(bins)/2 # breaks from bins
			prctBrks[0]=0; prctBrks[-1]=1.0 # end values
			print(prctBrks)
			# Interpolate break values
			BrkInterp=intp.interp1d(C,Hcntrs,kind='linear',
				bounds_error=False,fill_value=np.nan) # interpolation
			brks=BrkInterp(prctBrks) # interpolate break values
			brks[0]=Min; brks[-1]=Max # end values
			bins=BrkInterp(bins)
			prctBins=Cintrp(bins)
	elif mode in ['auto','kmeans']:
		nBins=bins # number of bins
		nBrks=nBins-1+2 # number of breaks
		bins=kmeans(I.reshape(-1,1),nBins)
		bins=bins[:,0]
		brks=np.zeros(nBrks) # empty array for break values
		brks[1:-1]=bins[:-1]+np.diff(bins)/2 # breaks from bins
		brks[0]=Min; brks[-1]=Max # end values
		prctBins=Cintrp(bins)
		prctBrks=Cintrp(brks)
		prctBrks[0]=C.min(); prctBrks[-1]=C.max()
	else:
		print('Mode not recognized')
	# Initial printouts
	if vocal is True:
		print('Mode: %s' % (mode))
		print('Bins: %i' % (nBins))
		print('Min: %.2f\tMax: %.2f' % (Min,Max))
	## Ouputs
	# Redefine image by breaks
	for i in range(nBrks-1):
		ndx=((I>=brks[i]) & (I<brks[i+1]))
		I_out[ndx]=np.mean(I[ndx])
		if vocal is True:
			print('\tbrks: %.2f - %.2f; bin: %.2f' \
				% (brks[i],brks[i+1],bins[i]))
	# Plot histogram
	if plot is True: 
		F=plt.figure() 
		# Histogram
		ax1=F.add_subplot(2,1,1)
		for i in range(nBrks): # break values
			ax1.axvline(brks[i],
				linestyle='-',color=(0.5,0.5,0.65),
				zorder=1)
		for j in range(nBins): # bin values
			binText='|%.0f' % (bins[j])
			ax1.text(bins[j],H.max(),binText,color='r')
		ax1.bar(Hcntrs,H,color='k',zorder=3) # histogram
		ax1.set_xlim([0,255])
		# CDF 
		ax2=F.add_subplot(2,1,2) 
		for i in range(nBrks): # break values
			ax2.axhline(prctBrks[i],
				linestyle='-',color=(0.5,0.5,0.65),
				zorder=1)
		for j in range(nBins):
			binText='._%.2f' % (prctBins[j])
			ax2.text(bins[j],prctBins[j],binText,color='r')
		ax2.plot(Hcntrs,C,color='k',zorder=3) # CDF
		ax2.set_xlim([0,255])
	return I_out 

# --- Remove tilt --- 
def untilt(data,dtype,ds=0,vocal=False,plot=False): 
	# INPUTS 
	#	data is the dataset, either as: 
	#	  m x 3 points (x,y,z) 
	#	  m x n 2D array with no spatial reference 
	#	dtype is the data type: 
	#	  'points' or 'pts' 
	#	  'image', 'img', or 'map' 
	#	ds is the downsample factor (power of 2) 
	#	  ... important for image data 
	#	vocal = True/False 
	#	plot  = True/False 
	# OUTPUTS 
	#	Tdata is the (un)tilted dataset 

	# Calculate normal to data points  
	Pln=fitPlane(data,dtype,ds=ds,vocal=vocal,plot=plot) 

	# Remove plane from data points 
	if dtype is 'points' or dtype is 'pts': 
		Tdata=data[:,2]-Pln.Q[:,2] # remove projected points 
	elif dtype is 'image' or dtype is 'img' or dtype is 'map': 
		Tdata=data-Pln.Z # remove hypothetical plane 

	# Print if specified 
	if vocal is True: 
		print('Removed ramp') 

	# Plot if specified 
	if plot is not False: 
		if dtype is 'points' or dtype is 'pts': 
			# Plot data as points 
			F=plt.figure() 
			ax=F.add_subplot(1,2,1,projection='3d') 
			ax.plot(data[:,0],data[:,1],data[:,2],'ko',zorder=3) 
			ax.set_zlim([data[:,2].min(),data[:,2].max()]); ax.set_zlabel('-z-') 
			ax.set_aspect(1) 
			ax.set_title('Orig points') 
			ax=F.add_subplot(1,2,2,projection='3d') 
			ax.plot(data[:,0],data[:,1],Tdata,'ko',zorder=3) 
			ax.set_zlim([data[:,2].min(),data[:,2].max()]); ax.set_zlabel('-z-') 
			ax.set_aspect(1) 
			ax.set_title('Tilted points') 
		elif dtype is 'image' or dtype is 'img' or dtype is 'map': 
			# Plot data as image 
			m,n=data.shape 
			if type(plot) is str:
				cmap=plot 
			else: 
				cmap='viridis' 
			F=plt.figure() 
			if m>n: 
				Prows=1; Pcols=2 
			else: 
				Prows=2; Pcols=1 
			ax1=F.add_subplot(Prows,Pcols,1) 
			cax1=ax1.imshow(data,cmap=cmap)
			ax1.set_title('Orig image') 
			F.colorbar(cax1,orientation='vertical') 
			ax2=F.add_subplot(Prows,Pcols,2) 
			cax2=ax2.imshow(Tdata,cmap=cmap) 
			ax2.set_title('Tilted image') 
			F.colorbar(cax2,orientation='vertical') 
	# Outputs 
	return Tdata 


###########################
### --- Compression --- ###
###########################

# --- SVD compression --- 
class svd_compress: 
	def __init__(self,I,k,vocal=False): 
		# Setup 
		m,n=I.shape # original image dimensions 
		self.orig_size=m*n # original image size 
		if vocal is True: 
			print('SVD compression: U.S.V^T') 
			print('\tOriginal dimensions: %i x %i' % (m,n)) 
		# Decomposition 
		U,s,VT=np.linalg.svd(I) 
		s=np.diag(s) # diagonal matrix 
		S=np.zeros((m,n)) # empty singular value matrix 
		S[:s.shape[0],:s.shape[1]]=s # value matrix 
		# Reduce dimensions 
		Uk=U[:,:k]   # left singular vectors 
		Sk=S[:k,:k]  # singular values 
		VTk=VT[:k,:] # right singular vectors 
		self.k=k    # record number of modes 
		self.U=Uk   # left singular vectors 
		self.S=Sk   # singular values 
		self.VT=VTk # right singular vectors 
		# Size comparison 
		size=Uk.shape[0]*Uk.shape[1] 
		size+=Sk.shape[0]*Sk.shape[1] 
		size+=VTk.shape[0]*VTk.shape[1] 
		self.size=size # final size 
		# Print statements if requested 
		if vocal is True: 
			print('\tU: ',Uk.shape) 
			print('\tS: ',Sk.shape) 
			print('\tVT:',VTk.shape) 
			print('\tOrig. size: %i' % (self.orig_size)) 
			print('\tFinal size: %i' % (size)) 
		if self.orig_size<size: 
			print('Warning: Compressed size > original size') 
	# Reconstitute if requested 
	def reconst(self): 
		# Note: This defeats the purpose of compression 
		self.I=self.U.dot(self.S.dot(self.VT)) 
	# Compute difference statistics 
	def compute_diffs(self,Iorig,vocal=False): 
		# Compare reconstituted image to original 
		if hasattr(self,'I'): 
			Ir=self.I # use reconst if exists 
		else: 
			Ir=self.U.dot(self.S.dot(self.VT)) # reconstitute 
		m,n=Ir.shape # image dimensions 
		assert Iorig.shape==Ir.shape # same dimensions 
		# Stats 
		self.abs_diff=np.sum(np.abs(Iorig-Ir)) # sum abs diff 
		self.pct_diff=self.abs_diff/(m*n) # sum perct diff 
		if vocal is True: 
			print('\t## STATS') 
			print('\tk modes: %i' % (self.k)) 
			print('\tSum abs. diff.: %f' % (self.abs_diff)) 
			print('\tSum pct. diff.: %f' % (self.pct_diff))		
	# Plot reconstituted image alone 
	def plot(self,cmap='viridis'): 
		if hasattr(self,'I'): 
			Ir=self.I # use reconst if exists 
		else: 
			Ir=self.U.dot(self.S.dot(self.VT)) # reconstitute 
		F=plt.figure() # new figure 
		ax=F.add_subplot(111) 
		cax=ax.imshow(Ir,cmap=cmap) 
		ax.set_xticks([]); ax.set_yticks([]) 
		F.colorbar(cax,orientation='vertical') 
		ax.set_title('k = %i modes' % (self.k)) 
	# Compare to original 
	def plot_compare(self,Iorig,cmap='viridis'): 
		# Reconstitute compressed image 
		if hasattr(self,'I'): 
			Ir=self.I # use reconst if exists 
		else: 
			Ir=self.U.dot(self.S.dot(self.VT)) # reconstitute 
		m,n=Ir.shape 
		assert Iorig.shape==Ir.shape # same dimensions 
		# Determine rows/cols 
		if m>n: 
			Prows=1; Pcols=3 
		else: 
			Prows=3; Pcols=1 
		# Plot 
		F=plt.figure() # new figure 
		ax1=F.add_subplot(Prows,Pcols,1) 
		ax1.imshow(Iorig,cmap=cmap) 
		ax1.set_xticks([]); ax1.set_yticks([]) 
		ax1.set_title('Orig') 
		ax2=F.add_subplot(Prows,Pcols,2) 
		ax2.imshow(Ir,cmap=cmap) 
		ax2.set_xticks([]); ax2.set_yticks([]) 
		ax2.set_title('%i comp' % (self.k)) 
		ax3=F.add_subplot(Prows,Pcols,3)
		cax3=ax3.imshow(Iorig-Ir,cmap='nipy_spectral') 
		ax3.set_xticks([]); ax3.set_yticks([]) 
		F.colorbar(cax3,orientation='horizontal')  
		ax3.set_title('Diff') 

# --- FFT compression --- 
class fft_compress: 
	# R is the compression ratio [0.0,1.0) 
	def __init__(self,I,R,vocal=False,plot=False): 
		# Check x,y dims are even numbers 
		m,n=I.shape 
		if m%2 is not 0: 
			I=I[:-1] # make even 
		if n%2 is not 0: 
			I=I[:-1] # make even 
		# Setup 
		m,n=I.shape # original image dimensions (even) 
		self.orig_size=m*n # original image size 
		# Fourier transform 
		IF=np.fft.fft2(I) # FFT 
		IF=np.fft.fftshift(IF) # center low freqs 
		# Reduced dimensions 
		m2=m/2; n2=n/2 # middle values 
		Rm2=int(R*m2); Rn2=int(R*n2) # reduced values 
		# Reduce to smaller array 
		IFr=IF[Rm2:-Rm2,Rn2:-Rn2] # dimensions to keep 
		R=np.fft.fftshift(IFr) # shift low freq back to outside 
		self.R=R # add to object 
		# Final stats 
		mr,nr=R.shape # reduced dimensions 
		self.orig_size=m*n # original size 
		self.reduc_size=mr*nr # reduced size 
		# Vocal if requested 
		if vocal is True: 
			print('\tOriginal dimensions: %i x %i' % (m,n)) 
			print('\t\t%i px' % (self.orig_size))  
			print('\tReduced dimensions: %i x %i' % (mr,nr)) 
			print('\t\t%i px' % (self.reduc_size)) 
		# Plot spectrum if requested 
		if plot is True: 
			F=plt.figure() 
			ax1=F.add_subplot(1,2,1) 
			cax1=ax1.imshow(np.log10(np.abs(IF))) 
			F.colorbar(cax1,orientation='horizontal') 
			ax1.set_title('Orig spectrum\n(%.2e px)' % (self.orig_size)) 
			ax2=F.add_subplot(1,2,2) 
			cax2=ax2.imshow(np.log10(np.abs(IFr))) 
			F.colorbar(cax2,orientation='horizontal') 
			ax2.set_title('Reduc. spectrum\n(%.2e px)' % (self.reduc_size)) 
	def reconst(self): 
		# Transform back to spatial domain 
		I=np.fft.ifft2(self.R) # inverse FFT 
		self.I=I.real # add real component to object 


###############################
### --- Image Statistcs --- ###
###############################

# --- Basic statistics --- 
class imgStats:
	def __init__(self,I,pctmin=0,pctmax=100,vocal=False,hist=False): 
		# Check if masked array 
		try: 
			I=I.compressed() 
		except: 
			pass 
		# Convert to 1D array
		I=np.reshape(I,(1,-1)).squeeze(0) # 1D array 
		# Stats 
		self.min=np.min(I)	   # min 
		self.max=np.max(I)	   # max 
		self.mean=np.mean(I)	 # mean 
		self.median=np.median(I) # median 
		self.std=np.std(I)     # standard deviation 
		self.vmin,self.vmax=np.percentile(I,(pctmin,pctmax)) 
		# Print stats 
		if vocal is True: 
			print('Image stats:') 
			print('\tmin: %f, max: %f' % (self.min,self.max)) 
			print('\tmean: %f' % (self.mean)) 
			print('\tmedian: %f' % (self.median)) 
			print('\tvmin: %f, vmax: %f' % (self.vmin,self.vmax)) 
		# Histogram 
		if hist is not False: 
			if type(hist)==int: 
				nbins=hist 
			else: 
				nbins=50 
			# All values 
			H0,H0edges=np.histogram(I,bins=nbins) 
			H0cntrs=H0edges[:-1]+np.diff(H0edges)/2 
			# Clipped values 
			I=I[(I>=self.vmin) & (I<=self.vmax)] 
			H,Hedges=np.histogram(I,bins=nbins) 
			Hcntrs=Hedges[:-1]+np.diff(Hedges)/2 
			# Plot 
			plt.figure() 
			# Plot CDF 
			plt.subplot(2,1,1) 
			plt.axhline(pctmin/100,color=(0.5,0.5,0.5))
			plt.axhline(pctmax/100,color=(0.5,0.5,0.5)) 
			plt.plot(H0cntrs,np.cumsum(H0)/np.sum(H0),'k') 
			# Pad 
			H0cntrs=np.pad(H0cntrs,(1,1),'edge')
			H0=np.pad(H0,(1,1),'constant') 
			Hcntrs=np.pad(Hcntrs,(1,1),'edge')
			H=np.pad(H,(1,1),'constant') 
			# Plot PDF 
			plt.subplot(2,1,2)  
			plt.fill(H0cntrs,H0,color=(0.4,0.5,0.5),alpha=1,label='orig') 
			plt.bar(Hcntrs,H,color='r',alpha=0.5,label='new') 
			plt.legend() 

# --- Standard deviation --- 
def imgSD(I,w=3,ds=0): 
	ds=int(2**ds); I=I[::ds,::ds]  
	N=w**2 # points per cell 
	h=np.ones((w,w))/N 
	EX=sig.convolve2d(I,h,'same') 
	EX2=sig.convolve2d(I**2,h,'same') 
	Var=EX2-EX**2 
	Var[Var<0]=0 # correct rounding error 
	SD=np.sqrt(Var) 
	return SD 

# --- Variogram ---- 
def variogram(I,H,N=1000,BG=0,vocal=False,plotImg=False,plotVar=False):
	# INPUTS
	#	I is a 2D array of image values
	#	H is an integer or array of lag values (px)
	#	N is the (initial) number of samples
	#	BG is the background value to be ignored 
	#	vocal prints results 
	#	plotImg plots the samples on the image 
	#	plotVar plots the variograms 
	# OUTPUTS 
	#	SVAR, SCOV, SDIF are all objects with similar attributes 
	#		.lags are the lags in vector H (pixels) 
	#		.vals are the var/etc. values 
	#		.count are the nb of samples at each lag 
	#		.fx (if avail) is a parameterized function over which 
	#		  any positive (> 0) lag can be evaluated 

	# Basic parameters
	m,n=I.shape # image dimensions
	if type(H) is int:
		H=np.array([H]) # convert to array
	H=H[H>1] # compute nugget separately
	nH=len(H)+1 # number of lags
	H=np.pad(H,(1,0),'constant',constant_values=1) # pad front of H = 1 
	if vocal is True: 
		print('H',H)
	# Pick random sample points 
	# np.random.seed(0) 
	si1=np.random.randint(0+H.max(),m-H.max(),N) # rand pts in x dimension
	sj1=np.random.randint(0+H.max(),n-H.max(),N) # rand pts in y dimension 
	# Remove samples of null values
	u=I[si1,sj1]!=0 # valid indices
	si1=si1[u]; sj1=sj1[u] # trim to valid values
	nS1=sum(u) # update count 
	if vocal is True: 
		print('N samples (valid) %i' % nS1) 
	# Plot image 
	if plotImg is True:
		S=plt.figure()
		axS=S.add_subplot(111) 
		vmin,vmax=np.percentile(I,(1,99))
		caxS=axS.imshow(np.ma.array(I,mask=(I==BG)),
			cmap='Greys_r',vmin=vmin,vmax=vmax,zorder=1)
		for k in range(nS1):
			axS.add_patch(Circle((sj1[k],si1[k]),H.max(),
				facecolor='y',alpha=0.3,zorder=2))
		axS.plot(sj1,si1,'b+',zorder=4)
	## Image samples
	# Empty variables
	count=np.zeros(nH) # number of samples in bin 
	svar=np.zeros(nH) # variance 
	scov=np.zeros(nH) # covariance
	sdif=np.zeros(nH) # abs. diff.
	n=np.zeros(nH) # number of samples in bin
	# Compute nugget 
	vnug=[True,True,True,True,False,True,True,True,True] # bool index array
	for s in range(nS1):
		# Adjacent values
		ctr=I[si1[s],sj1[s]] # center of kernel
		adj=I[si1[s]-1:si1[s]+2,sj1[s]-1:sj1[s]+2] # kernel
		adj=adj.reshape(1,-1).squeeze(0)[vnug] # keep adjacent values
		svar[0]=svar[0]+np.sum((ctr-adj)**2) # variance
		scov[0]=scov[0]+np.sum((ctr*adj)) # covariance
		sdif[0]=sdif[0]+np.sum(np.abs(ctr-adj)) # abs diff
	count[0]=8*nS1 # update count
	# Compute variance 
	nS2max=30 
	for h in range(1,nH):
		for s in range(nS1):
			# Determine number of random points
			nS2=np.min([nS2max,2*np.pi*H[h]]) 
			# Pick array of azimuths
			A=np.arange(nS2)/(nS2)*2*np.pi 
			# Convert to x,y coordinates
			si2=si1[s]+(H[h]*np.cos(A)).astype(int)
			sj2=sj1[s]+(H[h]*np.sin(A)).astype(int) 
			# Remove samples of null values 
			v=I[si2,sj2]!=0 # valid indices 
			si2=si2[v]; sj2=sj2[v] # trim to valid values
			if plotImg is True:
				axS.plot(sj2,si2,marker='.',color=(0.6,0.6,0.6),
					linewidth=0,alpha=0.3,zorder=3)
			# Recompute running totals 
			svar[h]=svar[h]+np.sum((I[si1[s],sj1[s]]-I[si2,sj2])**2) # variance
			scov[h]=scov[h]+np.sum((I[si1[s],sj1[s]]*I[si2,sj2])) # covariance
			sdif[h]=sdif[h]+np.sum(np.abs(I[si1[s],sj1[s]]-I[si2,sj2])) # abs diff
			count[h]=count[h]+sum(v) # update count
	# Finalize image plot
	if plotImg is True:
		axS.set_aspect(1) 
	# Summarize statistics
	svar=svar/2/count # semivariance
	scov=scov/2/count # semicovariance
	sdif=sdif/2/count # semidifference
	if vocal is True: 
		print('Svar',svar)
		print('Scov',scov) 
		print('Sdif',sdif)
		print('Count',count)
	## Fit distributions 
	# Range parameter 
	r0=np.arange(H[1],H.max(),1) # range values 
	nr=len(r0) # number of range values 
	# Semivariance 
	#	Fvar = B1 (1-exp(-d/r0)) + B2 
	Fvar=lambda h,r0: (1-np.exp(-h/r0)) # generic function 
	X=np.ones((nH,2))	 # design matrix 
	Bhat=np.zeros((2,nr)) # empty coefficients 
	res=np.zeros(nr)	  # empty residuals 
	for r in range(nr): 
		X[:,0]=Fvar(H,r0[r]) # update design matrix
		XTX1XT=np.dot(np.linalg.inv(np.dot(X.T,X)),X.T) # inverse 
		Bhat[:,r]=np.dot(XTX1XT,svar) # estimated coefficients 
		yhat=Bhat[0,r]*Fvar(H,r0[r])+Bhat[1,r] # estimated function values 
		res[r]=np.sum(np.abs(svar-yhat)) # residuals 
	Bhat=Bhat[:,res==res.min()] # optimum coefficients 
	r0var=r0[res==res.min()]	# optimum range 
	svar_sill=(Bhat[0]+Bhat[1]).squeeze(0) # calculate sill 
	svar_hat=Bhat[0]*Fvar(H,r0var)+Bhat[1] # optimum function 
	if vocal is True: 
		print('PARAMETERS:\nVariance:\n\tsill: %f\trange: %f' % (Bhat[0]+Bhat[1],r0var))  
	# Semicovariance
	#	Fcov = C1 exp(-d/r0) + C2 
	Fcov=lambda h,r0: np.exp(-h/r0) # generic function 
	X=np.ones((nH,2))	 # design matrix 
	Chat=np.zeros((2,nr)) # empty coefficients 
	res=np.zeros(nr)	  # empty residuals 
	for r in range(nr): 
		X[:,0]=Fcov(H,r0[r]) # update design matrix 
		XTX1XT=np.dot(np.linalg.inv(np.dot(X.T,X)),X.T) # inverse 
		Chat[:,r]=np.dot(XTX1XT,scov) # estimated coefficients 
		yhat=Chat[0,r]*Fcov(H,r0[r])+Chat[1,r] # estimated function values 
		res[r]=np.sum(np.abs(scov-yhat)) # residuals 
	Chat=Chat[:,res==res.min()] # optimum coefficients 
	r0cov=r0[res==res.min()]	# optimum range 
	scov_sill=Chat[1].squeeze(0) # sill  
	scov_hat=Chat[0]*Fcov(H,r0cov)+Chat[1] # optimum function 
	if vocal is True:
		print('Covariance:\n\tsill: %f\trange: %f' % (Chat[1],r0cov))
	# Semidifference 
	Fdif=lambda h: (np.log(h)) # generic function 
	X=np.ones((nH,2))	 # design matrix 
	X[:,0]=Fdif(H) # update design matrix 
	XTX1XT=np.dot(np.linalg.inv(np.dot(X.T,X)),X.T) # inverse 
	Dhat=np.dot(XTX1XT,sdif) # estimated coefficients 
	yhat=Dhat[0]*Fdif(H)+Dhat[1] # estimated function values 
	res=np.sum(np.abs(sdif-yhat)) # residuals 
	sdif_hat=Dhat[0]*Fdif(H)+Dhat[1] # optimum function 
	## Variogram
	# Plot variogram
	if plotVar is True:
		V=plt.figure()
		# Plot semivariance
		axV=V.add_subplot(3,1,1) 
		axV.axvline(r0var,linestyle='--',
			color=(0.6,0.6,0.6),zorder=1) # range 
		axV.axhline(Bhat[0]+Bhat[1],linestyle='--',
			color=(0.6,0.6,0.6),zorder=1) # sill 
		axV.text(0,svar_sill-0.05*(svar.max()-svar.min()),
			'%.1f' % (svar_sill),verticalalignment='top') 
		axV.plot(H,svar,'go',zorder=2) # values 
		axV.plot(H,svar_hat,'k',zorder=3) # estimate 
		axV.set_ylabel('semi-var.') 
		# Plot semicovariance
		axC=V.add_subplot(3,1,2) 
		axC.axvline(r0cov,linestyle='--',
			color=(0.6,0.6,0.6),zorder=1) # range 
		axC.axhline(Chat[1],linestyle='--',
			color=(0.6,0.6,0.6),zorder=1) # sill 
		axC.text(0,scov_sill+0.05*(scov.max()-scov.min()),
			'%.1f' % (scov_sill)) 
		axC.plot(H,scov,'bo',zorder=2) # values 
		axC.plot(H,scov_hat,'k',zorder=3) # estimate  
		axC.set_ylabel('semi-cov.') 
		# Plot semidifference 
		axD=V.add_subplot(3,1,3) 
		axD.plot(H,sdif,'ro',zorder=2) # values 
		axD.plot(H,sdif_hat,'k',zorder=3) # estimate 
		axD.set_ylabel('semi-diff.') 
		axD.set_xlabel('lag distance (pixels)')
	## Outputs 
	# Create object to hold outputs/parameters 
	class Vgram:
		def __init__(self,name,lags,vals,count,function=None):
			self.name=name # parameter name 
			self.lags=lags # lag vector 
			self.vals=vals # value at each lag 
			self.fx=function # descriptive function 

	# Semivariance 
	SVAR=Vgram(name='semivariance',lags=H,vals=svar,count=count) 
	SVAR.fx=lambda h,B1=Bhat[0],B2=Bhat[1],r0var=r0var: B1*(1-np.exp(-h/r0var))+B2 
	SVAR.range=r0var # range 
	SVAR.sill=svar_sill # sill 

	# Semicovariance 
	SCOV=Vgram(name='semicovariance',lags=H,vals=scov,count=count) 
	SCOV.fx=lambda h,C1=Chat[0],C2=Chat[1],r0cov=r0cov: C1*np.exp(-h/r0cov)+C2 
	SCOV.range=r0cov # range 
	SCOV.sill=scov_sill # sill 

	# Semidifference 
	SDIF=Vgram(name='semidifference',lags=H,vals=sdif,count=count) 


	return SVAR, SCOV, SDIF 


########################
### --- Blending --- ###
########################

# --- Linear ramp to image edges --- 
def linearBlend(I1,I2,plot=False): 
	# Setup 
	assert I1.shape == I2.shape; Exception: 'I1 and I2 must be same size' 
	m,n=I1.shape # original shape 
	# d=np.sqrt(m**2+n**2) # diagonal dimension 
	# Dimensions and sampling grid 
	m2=int(m/2); n2=int(n/2) 
	x=np.linspace(-1,1,n) 
	y=np.linspace(-1,1,m) 
	X,Y=np.meshgrid(x,y)
	# Blending factor 
	D2=np.sqrt(X**2+Y**2)/np.sqrt(2) 
	D1=1-D2 
	# Blend image 
	B=I1*D1+I2*D2  
	# Plot if desired 
	if plot is True: 
		F=plt.figure() 
		extent=(-n2,n2,-m2,m2)
		ax=F.add_subplot(111) 
		ax.imshow(I1,cmap='Greys_r',extent=extent,zorder=1) 
		ax.axhline(0,zorder=2) # central y-axis 
		ax.axvline(0,zorder=2) # central x-axis 
		ax.imshow(D1,cmap='viridis',extent=extent,alpha=0.5,zorder=3) 
	return B 

# --- Gaussian ramp --- 
def gaussBlend(I1,I2,d='auto',c=None,plot=False): 
	# Setup 
	assert I1.shape == I2.shape; Exception: 'I1 and I2 must be same size' 
	m,n=I1.shape # original shape 
	m2=int(m/2); n2=int(n/2) # centers of map 
	# Kernel center 
	if c is not None: 
		cx=c[0]; cy=c[1] # user-specified 
	else: 
		cx=n2; cy=m2 # map center 
	# Kernel width 
	if d is not 'auto': 
		# d is the 3-sigma width of the blending mask 
		d=d/3 
	else: 
		d=min([m,n])/3 
	# Sampling grid 
	x=np.arange(n); y=np.arange(m) 
	x=gauss(x,cx,d) 
	y=gauss(y,cy,d) 
	# Blending factor 
	D1=y.reshape(-1,1).dot(x.reshape(1,-1)) 
	D1=D1-D1.min(); D1=D1/D1.max() 
	D2=1-D1 
	# Blend image 
	B=I1*D1+I2*D2 
	# Plot if desired 
	if plot is True: 
		F=plt.figure() 
		ax=F.add_subplot(111) 
		ax.imshow(I1,cmap='Greys_r',zorder=1) 
		ax.axhline(cy,zorder=2) # central y-axis 
		ax.axvline(cx,zorder=2) # central x-axis 
		ax.imshow(D1,cmap='viridis',alpha=0.3,zorder=3) 
	return B
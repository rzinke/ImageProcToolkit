import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 

# Fit a 2D plane to data using Singular Value Decomposition
class fitPlane:
	def __init__(self,data,dtype,dx=1,dy=1,ds=0,method='svd',outputAll=True,vocal=False,plot=False):
		''' 
		Fit a plane to the points 
		INPUTS 
			data is the dataset, either as: 
			  m x 3 points (x,y,z) 
			  m x n 2D array with no spatial reference 
			dtype is the data type: 
			  'points' or 'pts' 
			  'image', 'img', or 'map'
			For a regular grid:
			  dx is the sample size in x
			  dy is the sample size in y
			ds is the downsample factor (power of 2)
			method is the inversion algorithm
			  'svd' or 'lsq'
			outputAll appends all outputs to the object 
			vocal = True/False 
			plot  = True/False 
		OUTPUTS 
			V are the first two right singular vectors 
			N is the unit normal to the plane 
		'''

		if vocal is True: 
			print('Fitting plane to %s' % (dtype)) 
			print('\tdata dimensions:',data.shape) 
		# Center and format  
		if dtype is 'points' or dtype is 'pts': 
			# Full x,y,z coordinates specified 
			nData=data.shape[0] # number of data points 
			xcenter=np.mean(data[:,0]) 
			ycenter=np.mean(data[:,1]) 
			zcenter=np.mean(data[:,2]) 
			data[:,0]=data[:,0]-xcenter # remove centroid 
			data[:,1]=data[:,1]-ycenter 
			data[:,2]=data[:,2]-zcenter 
			xMin=data[:,0].min(); xMax=data[:,0].max() 
			yMin=data[:,1].min(); yMax=data[:,1].max() 
			zMin=data[:,2].min(); zMax=data[:,2].max() 
			x=np.linspace(xMin,xMax,5); y=np.linspace(yMin,yMax,5) 
			X,Y=np.meshgrid(x,y) # grid surface 
			P=data # assign directly to points array 
			# Downsample 
			ds=int(2**ds) # downsample factor 
			P=P[::ds,:] # downsample data 
			nData=P.shape[0] # final number of data points 
		elif dtype is 'image' or dtype is 'img' or dtype is 'map': 
			# Map-like 2D array - create mapping coordinates 
			m,n=data.shape # original shape 
			nData=m*n # total number of data points 
			xcenter=0.; ycenter=0. 
			zcenter=np.nanmean(data); data=data-zcenter # remove centroid 
			xMin=-np.floor(n/2); xMax=np.ceil(n/2) 
			yMin=-np.floor(m/2); yMax=np.ceil(m/2) 
			zMin=np.nanmin(data); zMax=np.nanmax(data) 
			x=np.linspace(xMin,xMax,n)*dx; y=np.linspace(yMin,yMax,m)*dy
			X,Y=np.meshgrid(x,y) # grid surface 
			# Format into [x,y,z] array 
			P=np.hstack([X.reshape(nData,1),Y.reshape(nData,1),data.reshape(nData,1)]) 
			# Downsample 
			ds=int(2**ds) # downsample factor 
			P=P[::ds,:] # downsample data 
			P=P[np.isnan(P[:,2])==0] # remove nans from decomp
			nData=P.shape[0] # final number of data points 
		if vocal is True: 
			print('\tCenter (%f, %f, %f)' % (xcenter,ycenter,zcenter)) 
			print('\tBounds\n\t\tx: %f, %f\n\t\ty: %f, %f\n\t\tz: %f, %f' % 
				(xMin,xMax,yMin,yMax,zMin,zMax)) 
			print('\tP size:',P.shape) 
		if method.lower() in ['svd']:
			# Decompose using SVD
			U,s,VT=np.linalg.svd(P) 
			S=np.zeros((nData,3)); S[:3,:3]=np.diag(s) 
			# Reduce to 2D
			S[2,2]=0. # null third dimension 
			Q=U.dot(S.dot(VT)) # project onto plane 
			# Vectors 
			V=VT[:2,:].T # first two right singular vectors 
			dotV=np.dot(V[:,0],V[:,1]) # dot product to confirm orthonormality 
			N=np.cross(V[:,0],V[:,1]) # normal to plane 
			N=N/np.linalg.norm(N) # confirm unit length 
			Z=-1/N[2]*((N[0]*X)+(N[1]*Y)) # plane 
			if vocal is True: 
				print('\tSingular vectors:') 
				print('\t\tV1: ', V[:,0].T) 
				print('\t\tV2: ', V[:,1].T) 
				print('\t\t\tdot: %f' % (dotV)) 
				print('\tNormal to plane:\n\t\tN:', N.T) 
				print('\t\t\tdot: %f; %f' % (np.dot(N,V[:,0]),np.dot(N,V[:,1])))
		elif method.lower() in ['lsq','least squares']:
			# Set up design matrix
			G=P[:,:2] # design matrix
			# Beta vector
			Beta=np.linalg.inv(G.T.dot(G)).dot(G.T).dot(P[:,2]) # invert
			# Normal vector
			A=Beta[0]; B=Beta[1]
			C=np.cross(np.array([A,0,0]),np.array(0,B,0))
			N=np.array(A,B,C) # normal to plane
			N=N/np.linalg.norm(N) # confirm unit length
			Z=A*X+B*Y # plane
		# Plot if specified 
		if plot is not False: 
			if dtype is 'points' or dtype is 'pts': 
				# Plot data as points 
				F=plt.figure() 
				ax=F.add_subplot(111,projection='3d') 
				ax.plot_surface(X,Y,Z,color='b',alpha=0.25,zorder=1) 
				ax.quiver(0,0,0,V[0,0],V[1,0],V[2,0],color='g') 
				ax.quiver(0,0,0,V[0,1],V[1,1],V[2,1],color='r') 
				ax.quiver(0,0,0,N[0],N[1],N[2],color='k')
				ax.plot(Q[:,0],Q[:,1],Q[:,2],'b.',zorder=3)
				for i in range(nData): 
					ax.plot([Q[i,0],P[i,0]],[Q[i,1],P[i,1]],[Q[i,2],P[i,2]],'b')
				ax.plot(P[:,0],P[:,1],P[:,2],'ko',zorder=3) 
				ax.set_xlim([xMin,xMax]); ax.set_xlabel('-x-') 
				ax.set_ylim([yMin,yMax]); ax.set_ylabel('-y-') 
				ax.set_zlim([zMin,zMax]); ax.set_zlabel('-z-') 
				#ax.set_aspect(1) 
				ax.set_title('SVD plane fit (centered)')
			elif dtype is 'image' or dtype is 'img' or dtype is 'map': 
				# Plot data as image 
				if type(plot) is str:
					cmap=plot 
				else: 
					cmap='viridis' 
				F=plt.figure() 
				if m>n: 
					Prows=1; Pcols=2 
				else: 
					Prows=2; Pcols=1 
				extent=(xMin,xMax,yMin,yMax) 
				ax1=F.add_subplot(Prows,Pcols,1) 
				cax1=ax1.imshow(data,cmap=cmap,extent=extent)
				ax1.set_title('Data (centered)') 
				F.colorbar(cax1,orientation='vertical') 
				ax2=F.add_subplot(Prows,Pcols,2) 
				cax2=ax2.imshow(Z,extent=extent) 
				ax2.set_title('SVD plane fit') 
				F.colorbar(cax2,orientation='vertical') 
		# Outputs 
		self.V=V # singular vectors 
		self.N=N # normal to plane 
		if outputAll is True:
			self.Q=Q # points on plane 
			self.Z=Z # hypothetical plane 
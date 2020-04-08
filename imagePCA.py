import numpy as np
import matplotlib.pyplot as plt 

class stackPCA:
	def __init__(self,stack,standardize=True,verbose=False):
		# Data basics
		self.stack=stack.copy() # original data stack
		self.K,self.M,self.N=self.stack.shape # data set dimensions
		# K is the number of bands
		# M is the NS extent of the map
		# N is the EW extent of the map

		# Reshape as 2D (M.N x K) array and standardize
		MN=self.M*self.N # 2D stack width
		Data=np.zeros((MN,self.K)) # empty 2D array
		for k in range(self.K):
			Data[:,k]=stack[k,:,:].flatten()
			if standardize is True:
				Data[:,k]=(Data[:,k]-Data[:,k].mean())/Data[:,k].std()
		print('Data array shape: {}'.format(Data.shape))

		# Compute covariance matrix
		Cov=np.cov(Data.T) # Data.T has shape (K x M.N); each row is a variable
		
		# Compute eigen basis
		eigVals,eigVecs=np.linalg.eig(Cov)

		# Report if requested
		if verbose is True:
			print('Covariance matrix: {}\n{}'.format(Cov.shape,Cov))
			print('Eigenvalues: {}'.format(eigVals))

		# Sort from largest to smallest eigenvalues
		sortNdx=np.argsort(eigVals)[::-1] # sorting indices
		eigVals=eigVals[sortNdx] # sort values
		eigVecs=eigVecs[:,sortNdx] # sort vectors by column
		self.eigVals=eigVals; self.eigVecs=eigVecs

		# Project into eigen coordinates
		projData=np.dot(Data,eigVecs)
		print(projData.shape)

		# Reshape projected data into image shapes
		PCstack=np.zeros(stack.shape)
		for k in range(self.K):
			PCstack[k,:,:]=projData[:,k].reshape(self.M,self.N)

		self.PCstack=PCstack


	def plotRaw(self,band_names=None,ds=0):
		# Formatting
		ds=int(2**ds) # downsample factor
		rootK=np.sqrt(self.K)
		cols=np.ceil(rootK) # figure columns
		rows=np.floor(rootK)
		rows=np.ceil((self.K-(cols*rows))/cols+rows) # figure rows
		pos=1 # initial position
		if not band_names:
			band_names=range(self.K)
		# Plot
		RawFig=plt.figure('RawData')
		for k in range(self.K):
			ax=RawFig.add_subplot(rows,cols,pos)
			ax.imshow(self.stack[k,::ds,::ds],cmap='Greys_r')
			ax.set_xticks([]); ax.set_yticks([])
			ax.set_title('Img: {}'.format(band_names[k]))
			pos+=1 # update position


	def plotRetention(self):
		RetFig=plt.figure()
		ax=RetFig.add_subplot(111)
		ax.bar(range(self.K),self.eigVals/np.sum(self.eigVals)*100,
			align='center',width=0.4)
		ax.set_ylabel('% variance explained')
		ax.set_title('Variance retention')
		xticklabels=['PC{}'.format(k) for k in range(self.K+1)]
		ax.set_xticklabels(xticklabels)


	def plotPCs(self,cmap='Greys_r',ds=0):
		# Formatting
		ds=int(2**ds) # downsample factor
		rootK=np.sqrt(self.K)
		cols=np.ceil(rootK) # figure columns
		rows=np.floor(rootK)
		rows=np.ceil((self.K-(cols*rows))/cols+rows) # figure rows
		# Plot
		PCsFig=plt.figure('ProjectedData')
		for k in range(self.K):
			pc=k+1 # subtract 1 for index value
			ax=PCsFig.add_subplot(rows,cols,pc)
			ax.imshow(self.PCstack[k,::ds,::ds],cmap=cmap)
			ax.set_xticks([]); ax.set_yticks([])
			ax.set_title('PC {}'.format(pc))
		PCsFig.suptitle('Projected data')


	def plotPC(self,PC,cmap='Greys_r',bg=None,ds=0):
		# Formatting
		ds=int(2**ds) # downsample factor
		pcNdx=PC-1 # index of slice

		PC=self.PCstack[pcNdx,::ds,::ds]
		if bg:
			PC=np.ma.array(PC,mask=(PC==bg))

		# Plot
		PCFig=plt.figure('PrincipalComponent')
		ax=PCFig.add_subplot(111)
		cax=ax.imshow(PC,cmap=cmap)
		ax.set_title('PC {}'.format(PC))
		PCFig.colorbar(cax,orientation='vertical')
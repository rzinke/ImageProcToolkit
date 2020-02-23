import numpy as np 

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
		Centroids=Centroids_new.copy()
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
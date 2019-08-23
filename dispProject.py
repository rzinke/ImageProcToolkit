import numpy as np
import matplotlib.pyplot as plt


# --- Heading to vector ---
def head2vect(heading,units='deg'):
	'''
	Heading relative to north
	'''

	# Convert to radians if in degrees
	if units.lower() in ['deg','degree','degrees']:
		heading=np.deg2rad(heading)

	# Vector components
	Vx=np.sin(heading)
	Vy=np.cos(heading)

	# Scale to unit length
	M=np.sqrt(Vx**2+Vy**2) # vector magnitude
	Vx/=M 
	Vy/=M 

	return Vx,Vy


# --- Project displacements ---
def projectDisplacements(EW,NS,projectionVector):
	'''
	'''
	# Size of input arrays
	assert EW.shape==NS.shape, 'Displacement fields must be same size'
	m,n=EW.shape

	# Scale projection vector to unit length
	V=projectionVector/np.sqrt(projectionVector[0]**2+projectionVector[1]**2)
	V=V.reshape(1,2)

	# Reshape coordinates into 2x(mn) matrix
	C=np.vstack((EW.reshape(1,-1),NS.reshape(1,-1)))

	# Project components along unit vector
	P=V.dot(C)

	# Reshape to original size
	P=P.reshape(m,n)
	return P 
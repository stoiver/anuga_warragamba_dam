"""
Script to run Warragamba Dam simulation
"""

import anuga
import numpy





#====================================
# Set up the domain
#====================================

domain = anuga.Domain('Warragamba_P208_UTM.tsh')

#geo_ref = anuga.Geo_reference(zone=56, xllcorner=277602, yllcorner=6248165)
#domain.set_georeference(geo_ref)

#====================================
# Read in the elevation data
#====================================

x = numpy.load('data/elevation/x.npy')
y = numpy.load('data/elevation/y.npy')
elev = numpy.load('data/elevation/elev.npy')

x = x.reshape((-1,1))
y = y.reshape((-1,1))
elev = elev.reshape((-1,1))

xyvalues = numpy.hstack((x,y,elev))

#data_points = numpy.hstack((x,y))
#georef = anuga.Geo_reference(zone=-1)
#geo_data = anuga.Geospatial_data(geo_reference=georef, data_points=data_points)

#print(geo_data)
#print(geo_data.geo_reference)

print('Max eastings ', numpy.max(x))
print('Min eastings ', numpy.min(x))
print('Max northings ', numpy.max(y))
print('Min northings ', numpy.min(y))

from anuga.utilities.quantity_setting_functions import make_nearestNeighbour_quantity_function

print('Start making elev_fun')
elev_fun = make_nearestNeighbour_quantity_function(xyvalues,domain)

print('set_quantity')
domain.set_quantity('elevation', function=elev_fun, location='centroids')

Br = anuga.Reflective_boundary(domain)

domain.set_boundary({'exterior': Br, 'Dam' : Br})

for t in domain.evolve(yieldstep=0.01, finaltime=0.0):
    domain.print_timestepping_statistics()


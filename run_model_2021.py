"""
Script to run Warragamba Dam simulation
"""

import anuga
from anuga.operators.set_stage_operator import Polygonal_set_stage_operator
import numpy
import pandas

#====================================
# Setup some parameters
#====================================
max_area = 40000
show_plots = False

#====================================
# Set up the domain
#====================================

#domain = anuga.Domain('Warragamba_P208_UTM_clean.tsh')

dam_wall_poly = numpy.loadtxt('dam_wall.csv', delimiter=',')
riverwalls = {'dam_wall' : dam_wall_poly}

boundary_poly = numpy.loadtxt('boundary_poly.csv', delimiter=',')
boundary_tags={'Dam':  [i for i in range(754,760)]}

below_dam_poly = [ [point[0], point[1]] for point in dam_wall_poly] + [boundary_poly[i] for i in range(760,753, -1)]

mesh_filename = 'warragamba_from_regions_'+str(max_area)+'.tsh'

if anuga.myid == 0:

    domain = anuga.create_domain_from_regions(boundary_poly, 
                                            boundary_tags,
                                            maximum_triangle_area = max_area,
                                            breaklines = riverwalls.values(),
                                            mesh_filename=mesh_filename)

    domain.set_name('domain_2021', timestamp=True)
    domain.set_flow_algorithm('DE1')
    domain.set_low_froude(1)

    domain.print_statistics()
    #geo_ref = anuga.Geo_reference(zone=56, xllcorner=277602, yllcorner=6248165)
    #domain.set_georeference(geo_ref)

    #====================================
    # Read in the elevation data
    #====================================

    x = numpy.load('data/elevation/x.npy')
    y = numpy.load('data/elevation/y.npy')
    elev = numpy.load('data/elevation/elev.npy')

    # Add in some points downstream of dam
    xd = boundary_poly[754:761,0]
    yd = boundary_poly[754:761,1]
    ed = numpy.ones_like(xd)*40

    xdd = xd - 50.0
    ydd = yd - 130.0
    edd = numpy.ones_like(xd)*40

    x = numpy.append(x,xd)
    y = numpy.append(y,yd)
    elev = numpy.append(elev,ed)

    x = numpy.append(x,xdd)
    y = numpy.append(y,ydd)
    elev = numpy.append(elev,edd)

    x = x.reshape((-1,1))
    y = y.reshape((-1,1))
    elev = elev.reshape((-1,1))

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    print('Try to plot 30M points')


    if show_plots:
        plt.figure(1)
        index = numpy.random.choice(x.shape[0], 100000, replace=False)
        X = x[index]
        Y = y[index]
        Z = elev[index]
        plt.scatter(X,Y, c=Z, marker='.')
        plt.colorbar()
        plt.show()

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

    print('Making elev_fun (This can take some time)')
    elev_fun = make_nearestNeighbour_quantity_function(xyvalues, domain, method='min', k_nearest_neighbours=10)


    print('set_quantity')
    #domain.set_quantity('elevation', function=elev_fun, location='centroids')
    Elev = domain.quantities['elevation']
    Elev.set_values(function=elev_fun, location='centroids')

    # Find the triangles below the dam
    ids = domain.get_triangles_inside_polygon(below_dam_poly)
    Elev.set_values(40.0, indices=ids)

    if show_plots:
        plt.figure(1)
        X = domain.centroid_coordinates[:,0]
        Y = domain.centroid_coordinates[:,1]
        Z = Elev.centroid_values
        plt.scatter(X,Y, c=Z, marker='.')
        plt.colorbar()
        plt.show()

    Stage = domain.quantities['stage']
    Stage.set_values(116.0, location='centroids')

    # Hard code stage to elevation down stream of dam
    #print([7441, 7452, 7453, 7455, 7456, 7446, 7451, 7449, 7450, 7424, 7433])
    print(ids)

    Stage.set_values(40.0, indices=ids)
    Stage.maximum(Elev)

    if show_plots:
        plt.figure(1)
        X = domain.centroid_coordinates[:,0]
        Y = domain.centroid_coordinates[:,1]
        Z = Stage.centroid_values
        plt.scatter(X,Y, c=Z, marker='.')
        plt.colorbar()
        plt.show()

    domain.set_quantity('friction', 0.03)

    print(numpy.max(Elev.centroid_values))
    print(numpy.min(Elev.centroid_values))
    print(numpy.max(Elev.vertex_values))
    print(numpy.min(Elev.vertex_values))

else:
    domain = None

#=================================
# Distribute domain
#=================================
domain = anuga.distribute(domain)

#=================================
# Setup inflow and outflow via inlet_operators
#=================================

# Data: [ Filename, lat, long, inlet radius, type, default flows ]
station_data = {212241: ['Warragamba Wier 2021.xlsx', -33.87694444, 150.6063889, 100.0, 'outflow', (0.0, 0.0)],
            212250: ['212250_Cox_River_Kelpie_Pt_2021.xlsx', -33.87124191, 150.2540137, 100.0, 'inflow', (10.0, 10.0)],
            212270: ['212270_Wollondilly_River_Jooriland_2021.xlsx', -34.22765922, 150.2527921, 1000.0, 'inflow', (10.0, 10.0)],
            212280: ['212280_Nattai_River_Causeway_2021.xlsx', -34.14555579, 150.4247247, 1000.0, 'inflow', (2.0, 2.0)],
            212260: ['212260_Kowmung_Cedar_ford_2021.xlsx', -33.94805556, 150.2430556, 8000.0, 'inflow', (10.0, 10.0)],
            212243: ['212243_Warragamba_Dam_2021.xlsx', -33.89111111, 150.5911111, 200.0, 'dam', (0.0, 0.0)]}

import utm

inflow_operators = []
for key, value in station_data.items():

    if value[4] == 'inflow':
        if anuga.myid ==0: print('Setup inflow station ',key)
        lat = value[1]
        long =value[2]
        u = utm.from_latlon(lat,long)
        easting = u[0]
        northing = u[1]
        radius = value[3]
        station_filename = 'data/flows_2021/'+value[0]

        df = pandas.read_excel(station_filename)
        df['Time'] = df['Time'] - df['Time'][0]
        station_time = numpy.array(df['Time'])*3600*24    # convert from day to seconds

        if anuga.myid ==0: print(numpy.max(station_time))
        if anuga.myid ==0: print(numpy.min(station_time))
        station_Q    = numpy.array(df['Discharge'])/36/24*10 # convert from ML/day to m^3/sec
        from scipy import interpolate

        station_Q_fun = interpolate.interp1d(station_time, station_Q, fill_value=value[5], bounds_error=False)

        t = numpy.linspace(-3600*24,3600*24*10, 1000)
        q = station_Q_fun(t)


        #df.plot('Time', 'Discharge')
        #plt.show()

        if anuga.myid == 0 and show_plots:
            plt.figure(1)
            plt.plot(t,q)
            plt.title("Station "+str(key))
            plt.show()


        station_region = anuga.Region(domain, center=[easting, northing], radius=radius)
        inflow_operators.append(anuga.Inlet_operator(domain, station_region, Q=station_Q_fun, label=str(key), verbose=True))

        if inflow_operators[-1] is not None:
            inflow_operators[-1].print_statistics()

#==================================
# Setup simple BC
#==================================
Br = anuga.Reflective_boundary(domain)
Bd = anuga.Dirichlet_boundary([40.0, 0.0,0.0])

domain.set_boundary({'exterior': Br, 'Dam' : Bd})

#===================================
# Setup Dam via riverwall
#===================================
domain.riverwallData.create_riverwalls(riverwalls, verbose=True)

#===================================
# Setup gauge Points
#===================================
gauge_points = [[248288, 6251012.0],
                [254752, 6219030],
                [267288, 6239628],
                [277487, 6248067]]

print("Gauge_IDs")
gauge_ids = []
for gauge_point in gauge_points:
    try:
        gauge_id = domain.get_triangle_containing_point(gauge_point)
        gauge_ids.append(gauge_id)
        print(anuga.myid, gauge_id)
    except:
        print('couldnt find ', gauge_point, ' on proc ', anuga.myid)


day = 24*3600
hr = 3600
min = 60
sec = 1

Stage_c = domain.quantities['stage'].centroid_values

# Set a negative starttime to allow a burn in phase
domain.set_starttime(0*hr)

initial_water_volume = domain.get_water_volume()

import sys

for t in domain.evolve(yieldstep=10*min, finaltime=10*day):
    if anuga.myid ==0: 
        print(72*"=")
        domain.print_timestepping_statistics(time_unit='hr')
        print(72*"=", flush=True)

    anuga.barrier()

    if anuga.myid ==0: print("Current inflows")
    sys.stdout.flush() 
    
    anuga.barrier()

    total_inflow = 0.0
    for inflow_operator in inflow_operators:
        if inflow_operator is not None:
            # Only accumulate total_inflow from master
            if inflow_operator.master_proc == anuga.myid:
                inflow_operator.print_timestepping_statistics()
                sys.stdout.flush() 
                total_inflow += inflow_operator.get_total_applied_volume()

    # Send all the total_inflows to process 0
    if anuga.myid == 0:

        for i in range(1,anuga.numprocs):
            total_inflow += anuga.receive(i)
    else:
        anuga.send(total_inflow, 0)

    anuga.barrier()

    for i in range(anuga.numprocs):
        if anuga.myid == i:
            print('On P%g '%i, Stage_c[gauge_ids], flush=True)
            sys.stdout.flush() 
        anuga.barrier()

    water_volume  = domain.get_water_volume()
    boundary_flux = domain.get_boundary_flux_integral()
    water_volume_added = domain.get_water_volume() - initial_water_volume

    check = initial_water_volume + total_inflow + boundary_flux - water_volume

    if anuga.myid == 0: print('Domain water volume ', water_volume)
    if anuga.myid == 0: print('Boundary flux integral ', boundary_flux)
    if anuga.myid == 0: print('Water volume added ', water_volume_added)
    if anuga.myid == 0: print('Total inflow volume ', total_inflow)
    if anuga.myid == 0: print('Initial Volume + Total inflow + Boundary flux - Domain volume ', check)


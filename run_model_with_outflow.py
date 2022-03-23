"""
Script to run Warragamba Dam simulation
"""

import anuga
import numpy
import pandas
import matplotlib as mpl
import matplotlib.pyplot as plt

#====================================
# Setup some parameters
#====================================
#flow_data = 'flows_1990'
#flow_data = 'flows_2020'
flow_data = 'flows_1989'

max_area = 10000
#max_area = 40000


show_plots = True
print_inlet_data = False
calc_elevation = False
dam_height = 40.0
reservoir_stage = 115.831

#====================================
# Set up the domain
#====================================

#domain = anuga.Domain('Warragamba_P208_UTM_clean.tsh')

dam_wall_polyline = numpy.loadtxt('data/dam_wall.csv', delimiter=',')
# drop the wall height from the file
dam_wall_polyline = [ [point[0], point[1]] for point in dam_wall_polyline]

# now introduce new dam_height
dam_wall_riverwall =  [ [point[0], point[1], dam_height ] for point in dam_wall_polyline]
riverwalls = {'dam_wall' : dam_wall_riverwall}


boundary_poly = numpy.loadtxt('data/boundary_poly.csv', delimiter=',')
boundary_tags={'Dam':  [i for i in range(754,760)]}

below_dam_polygon = dam_wall_polyline + [boundary_poly[i] for i in range(760,753, -1)]

mesh_filename = 'warragamba_from_regions_'+str(max_area)+'.tsh'

if anuga.myid == 0:

    domain = anuga.create_domain_from_regions(boundary_poly, 
                                            boundary_tags,
                                            maximum_triangle_area = max_area,
                                            breaklines = riverwalls.values(),
                                            mesh_filename=mesh_filename)

    domain.set_name('domain_'+flow_data, timestamp=True)
    domain.set_flow_algorithm('DE1')
    domain.set_low_froude(1)

    domain.print_statistics()
    #geo_ref = anuga.Geo_reference(zone=56, xllcorner=277602, yllcorner=6248165)
    #domain.set_georeference(geo_ref)

    #====================================
    # Calculate the elevation quantity
    # If calc_elevation is True then 
    # read in the large data files (250MB)
    # from the initial_data directory, 
    # otherwise read in the array data
    # interpolated to the domain. Interpolated 
    # array available for domain with max_area 
    # set to 10000 and 40000
    #====================================
    if calc_elevation:
        x = numpy.load('initial_data/elevation/x.npy')
        y = numpy.load('initial_data/elevation/y.npy')
        elev = numpy.load('initial_data/elevation/elev.npy')

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

        if show_plots:
            print(50*'-')
            print('Plot selection of 30M points')
            print(50*'-')
            plt.figure(1)
            index = numpy.random.choice(x.shape[0], 100000, replace=False)
            X = x[index]
            Y = y[index]
            Z = elev[index]
            plt.scatter(X,Y, c=Z, marker='.')
            plt.title('Elevation')
            plt.colorbar()
            plt.show()

        xyvalues = numpy.hstack((x,y,elev))

        print('Max eastings ', numpy.max(x))
        print('Min eastings ', numpy.min(x))
        print('Max northings ', numpy.max(y))
        print('Min northings ', numpy.min(y))

        from anuga.utilities.quantity_setting_functions import make_nearestNeighbour_quantity_function

        print(50*'-')
        print('Interpolating elevation to domain (This can take some time)')
        print(50*'-')
        elev_fun = make_nearestNeighbour_quantity_function(xyvalues, domain, method='min', k_nearest_neighbours=10)

        #domain.set_quantity('elevation', function=elev_fun, location='centroids')
        Elev = domain.quantities['elevation']
        Elev.set_values(function=elev_fun, location='centroids')

        elev_c = Elev.centroid_values

        elev_filename = 'data/elev_c_'+str(max_area)+'.npy'
        numpy.save(elev_filename, elev_c)

    else:
        elev_filename = 'data/elev_c_'+str(max_area)+'.npy'
        elev_c = numpy.load(elev_filename)

        Elev = domain.quantities['elevation']
        Elev.set_values(elev_c, location='centroids')


    #data_points = numpy.hstack((x,y))
    #georef = anuga.Geo_reference(zone=-1)
    #geo_data = anuga.Geospatial_data(geo_reference=georef, data_points=data_points)

    #print(geo_data)
    #print(geo_data.geo_reference)



    # Find the triangles below the dam
    ids = domain.get_triangles_inside_polygon(below_dam_polygon)
    Elev.set_values(40.0, indices=ids)

    if show_plots:
        plt.figure(1)
        X = domain.centroid_coordinates[:,0]
        Y = domain.centroid_coordinates[:,1]
        Z = Elev.centroid_values
        plt.scatter(X,Y, c=Z, marker='.')
        plt.title('Domain elevation')
        plt.colorbar()
        plt.show()

    Stage = domain.quantities['stage']
    Stage.set_values(reservoir_stage, location='centroids')

    # Hard code stage to elevation down stream of dam
    #print([7441, 7452, 7453, 7455, 7456, 7446, 7451, 7449, 7450, 7424, 7433])
    #print(ids)

    # Set Stage below dam
    #Stage.set_values(40.0, indices=ids)

    Stage.maximum(Elev)

    if show_plots:
        plt.figure(1)
        X = domain.centroid_coordinates[:,0]
        Y = domain.centroid_coordinates[:,1]
        Z = Stage.centroid_values
        plt.scatter(X,Y, c=Z, marker='.')
        plt.title('Domain stage')
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
station_data = {212241: ['212241.xlsx', -33.87694444, 150.6063889, 100.0, 'outflow', (0.0, 0.0)],
            212250: ['212250.xlsx', -33.87124191, 150.2540137, 100.0, 'inflow', (10.0, 10.0)],
            212270: ['212270.xlsx', -34.22765922, 150.2527921, 1000.0, 'inflow', (10.0, 10.0)],
            212280: ['212280.xlsx', -34.14555579, 150.4247247, 1000.0, 'inflow', (2.0, 2.0)],
            212260: ['212260.xlsx', -33.94805556, 150.2430556, 8000.0, 'inflow', (10.0, 10.0)],
            212243: ['212243.xlsx', -33.89111111, 150.5911111, 200.0, 'dam', (0.0, 0.0)]}

import utm

flow_operators = []

start_time = 1.0e100

for key, value in station_data.items():

    if value[4] == 'inflow':
        if anuga.myid ==0: print('Setup inflow station ',key)
        lat = value[1]
        long =value[2]
        u = utm.from_latlon(lat,long)
        easting = u[0]
        northing = u[1]
        radius = value[3]
        station_filename = 'data/'+flow_data+'/'+value[0]

        df = pandas.read_excel(station_filename)
        #df['Time'] = df['Time'] - df['Time'][0]

        station_time = numpy.array(df['Time'])*3600*24    # convert from day to seconds
        start_time = min(station_time[0], start_time)

        if anuga.myid ==0: print(numpy.max(station_time))
        if anuga.myid ==0: print(numpy.min(station_time))
        station_Q    = numpy.array(df['Discharge'])/(24*3600.0)*1000.0 # convert from ML/day to m^3/sec
        from scipy import interpolate

        station_Q_fun = interpolate.interp1d(station_time, station_Q, fill_value=value[5], bounds_error=False)

        if anuga.myid == 0 and show_plots:
            t = numpy.linspace(start_time-3600*24,start_time+3600*24*10, 1000)
            q = station_Q_fun(t)
            plt.figure(1)
            plt.plot(t,q)
            plt.title("Station "+str(key))
            plt.xlabel('Time (s)')
            plt.ylabel('Flow (m^3/s)')
            plt.show()


        outflow_region = anuga.Region(domain, center=[easting, northing], radius=radius)
        flow_operators.append(anuga.Inlet_operator(domain, outflow_region, Q=station_Q_fun, label=str(key), verbose=True))

        if flow_operators[-1] is not None:
            flow_operators[-1].print_statistics()

    if value[4] == 'outflow':
        if anuga.myid ==0: print('Setup outflow station ',key)
        lat = value[1]
        long =value[2]
        u = utm.from_latlon(lat,long)
        easting = u[0]
        northing = u[1]
        radius = value[3]
        station_filename = 'data/'+flow_data+'/'+value[0]

        df = pandas.read_excel(station_filename)
        #df['Time'] = df['Time'] - df['Time'][0]
        station_time = numpy.array(df['Time'])*3600*24    # convert from day to seconds
        start_time = min(station_time[0], start_time)

        if anuga.myid ==0: print(numpy.max(station_time))
        if anuga.myid ==0: print(numpy.min(station_time))
        station_Q    = numpy.array(df['Discharge']) # already m^3/sec
        from scipy import interpolate

        outflow_Q_fun = interpolate.interp1d(station_time, station_Q, fill_value=value[5], bounds_error=False)

        Q_outflow = lambda t : -outflow_Q_fun(t)

        if anuga.myid == 0 and show_plots:
            t = numpy.linspace(start_time-3600*24,start_time+3600*24*10, 1000)
            q = Q_outflow(t)
            plt.figure(1)
            plt.plot(t,q)
            plt.xlabel('Time (s)')
            plt.ylabel('Flow (m^3/s)')
            plt.title("Station "+str(key))
            plt.show()


        station_region = anuga.Region(domain, polygon = below_dam_polygon)
        flow_operators.append(anuga.Inlet_operator(domain, station_region, Q=Q_outflow, label=str(key), verbose=True))

        if flow_operators[-1] is not None:
            flow_operators[-1].print_statistics()

#==================================
# Setup simple BC
#==================================
Br = anuga.Reflective_boundary(domain)
Bd = anuga.Dirichlet_boundary([40.0, 0.0,0.0])

domain.set_boundary({'exterior': Br, 'Dam' : Br})

#===================================
# Setup Dam via riverwall
#===================================
domain.riverwallData.create_riverwalls(riverwalls, verbose=True)


anuga.barrier()
#===================================
# Setup gauge Points
#===================================
gauge_points = [[248288, 6251012.0],
                [254752, 6219030],
                [267288, 6239628],
                [277487, 6248067]]

if anuga.myid == 0:
    print(80*"=")
    print("Gauge IDs")
    print(80*"=", flush=True)

anuga.barrier()

gauge_tids = []
gauge_ids = []
for i, gauge_point in enumerate(gauge_points):
    try:
        gauge_tid = domain.get_triangle_containing_point(gauge_point)
        gauge_tids.append(gauge_tid)
        gauge_ids.append(i)
        print(str(i)+'-th gauge point ', gauge_point, ' found in triangle ', gauge_tid, ' on process ', anuga.myid)
    except: 
        pass
        #print('couldnt find ', gauge_point, ' on proc ', anuga.myid)

anuga.barrier()

# Tell process 0 where the gauge points located


if anuga.myid == 0:
    gauge_processes = { 0 : gauge_ids}
    for iproc in range(1, anuga.numprocs):
        gauge_processes[iproc] = anuga.receive(iproc)
else:
    anuga.send(gauge_ids, 0)

anuga.barrier()

if anuga.myid == 0:
    print(gauge_processes)


day = 24*3600
hr = 3600
min = 60
sec = 1

Stage_c = domain.quantities['stage'].centroid_values

# start_time is set to the earliest time in the flow data files.
# set an earlier time if you want to have a "burn in" period
domain.set_starttime(start_time)

initial_water_volume = domain.get_water_volume()

import sys

for t in domain.evolve(yieldstep=10*min, duration=10*day):
    if anuga.myid ==0: 
        print()
        print(80*"=")
        domain.print_timestepping_statistics(time_unit='day')
        print('Evolution Time: ',domain.relative_time, '(s)')
        print(80*"=", flush=True)

    #anuga.barrier()

    #if anuga.myid ==0: print("Current inflows")
    #sys.stdout.flush() 
    
    anuga.barrier()

 
    total_inflow = 0.0
    for inflow_operator in flow_operators:
        if inflow_operator is not None:
            # Only accumulate total_inflow from master
            if inflow_operator.master_proc == anuga.myid:
                if print_inlet_data: 
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

    # Move the gauge point stage data to process 0 for printing
    if anuga.myid == 0:
        gauge_stage = [ Stage_c[gauge_tid] for gauge_tid in gauge_tids]
        gauge_stages = {0 : gauge_stage }
        for iproc in range(1, anuga.numprocs):
            gauge_stages[iproc] = anuga.receive(iproc)
    else:
        gauge_stage = [ Stage_c[gauge_tid] for gauge_tid in gauge_tids]
        anuga.send(gauge_stage, 0)

    anuga.barrier()

    # if anuga.myid == 0:
    #     print(gauge_stages)
    #     print(gauge_processes)

    if anuga.myid == 0:
        gauge_output = 4*[0]
        for iproc in gauge_processes:
            stages = gauge_stages[iproc]
            gauges = gauge_processes[iproc]
            #print(gauges, stages)
            for (gauge_id, stage_v) in zip (gauges, stages):
                gauge_output[gauge_id] = stage_v

    if anuga.myid == 0:
        print(50*'-')
        print('Stage at Gauge Points')
        print(50*'-')
        print(gauge_output)
        print(' ')
            


    water_volume  = domain.get_water_volume()
    boundary_flux = domain.get_boundary_flux_integral()
    water_volume_added = domain.get_water_volume() - initial_water_volume

    check = initial_water_volume + total_inflow + boundary_flux - water_volume

    if anuga.myid == 0: 
        print(50*'-')
        print('Flow Statisitics')
        print(50*'-')
        print('Domain water volume (m^3):    ', water_volume)
        print('Boundary flux integral (m^3): ', boundary_flux)
        print('Water volume added (m^3):     ', water_volume_added)
        print('Total inflow volume (m^3):    ', total_inflow)
        print('Initial Volume + Total inflow + Boundary flux - Domain volume (m^3): ', check)


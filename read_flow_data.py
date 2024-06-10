
def read_flow_data(key, value, data_dir, data_base_timestamp):
    """
    Read inflow and outflow and dam height data
    
    :param station_datum: Info about data

    Example:

    key: 212241
    value: ['212241.xlsx', -33.87694444, 150.6063889, 100.0, 'outflow', (0.0, 0.0)]

    consisting of the data file, location lat, long, radius: 100.0, type of data set:[outflow, inflow, dam], limits for interpolation, unit_factor
    """

    import utm
    import pandas
    import numpy

    station = {}


    if value[4] == 'inflow'  or value[4] == 'outflow' or value[4] == 'dam' :
        print('Setup '+value[4]+' station ',key)


        station['type'] = value[4]

        #unit_factor = 1000.0/(3600*24)
        unit_factor = value[6]
        unit_name = value[7]

        lat = value[1]
        long =value[2]
        u = utm.from_latlon(lat,long)
        easting = u[0]
        northing = u[1]
        
        station['utm'] = [easting, northing]
        
        radius = value[3]
        station['filename'] = data_dir+value[0]
        
        station['key'] = key

        df = pandas.read_excel(station['filename'])
        #df['Time'] = df['Time'] - df['Time'][0]
        station_time = numpy.array(df['Time'])*3600*24    # convert from day to seconds
        station_time = station_time + data_base_timestamp # convert to seconds from 1/1/1970 UTC

        station['time_series'] = station_time

        #print(numpy.max(station_time))
        #print(numpy.min(station_time))
        station_Q    = numpy.array(df[unit_name])*unit_factor # m^3/sec
        from scipy import interpolate, integrate

        station_Q_fun = interpolate.interp1d(station_time, station_Q, fill_value=value[5], bounds_error=False)
        station['Q_fun'] = station_Q_fun

        station_flow = numpy.zeros_like(station_time)
        for k, finaltime in enumerate(station_time):
            if k == 0:
                pass
            else:
                station_flow[k], err = integrate.quad(station_Q_fun, station_time[k-1], station_time[k], limit=1000, epsabs=100.0)

        station['flow'] = station_flow
        

    return station


# Plot the perimeter of Warragamba Dam

import os
import pandas
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings


plt.ion()


import os
year = os.path.split(os.getcwd())[1][-4:]

stations = { '212250': 'Coxs River at Kelpie Point',
             '212260': 'Kowmung River at Cedar Ford',
             '212243': 'Warragamba Dam',
             '212270': 'Wollondilly River at Jooriland',
             '212280': 'Nattai River Causeway',
             '212241': 'Warragamba Weir'}

fig_n = 0

for station_id, description in stations.items():
    
    try:
        filename = station_id + '.xlsx'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pandas.read_excel(filename)
    except:
        filename = station_id + '.xls'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pandas.read_excel(filename)        

    #df['Time'] = df['Time'] - df['Time'][0]

    for column in df.columns:
        if column == "Time": continue

        fig_n += 1
        plt.figure(fig_n)
        df.plot.line(x = 'Time', y = column)

        plt.xlabel('Day')
        if column == 'Depth':
            plt.ylabel('depth (m)')
        if column == 'Discharge':
            plt.ylabel('discharge (m^3/s)')
    
        plt.title(description)
        plt.show()
        plt.savefig(station_id +'_' + column + '.png')
        print('Creating File '+station_id+'_'+column+'.png')

    

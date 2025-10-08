from GISAXS_Analysis import GISAXS_setup_v21 as g
from GISAXS_Analysis import Graphing_Analysis as graphing_analysis

'''
exp_data_directory = r'C:\BornAgainSimulations\data\tif'
    
exp_filenames = ['Mica_0p2deg.tif','Quartz_0p2deg.tif','Sapphire_0p2deg.tif','SiN_0p2deg.tif']

for exp_filename in exp_filenames:
    g.tifToNpzConversion(exp_filename, exp_data_directory, 'feb', 0.2)

exp_data_directory = r'C:\BornAgainSimulations\data\tif'
    
exp_filename_Si = 'Si_0p1or0p2_deg.tif'


g.tifToNpzConversion(exp_filename_Si, exp_data_directory, 'dec', 0.2)
'''

exp_data_directory = r'C:\BornAgainSimulations\data\tif'

exp_filename_Si1 = 'Si_10deg.tif'

exp_filename_Si2 = 'Mica_15deg.tif'

#exp_filename_Si3 = 'Quartz_10deg.tif'
#exp_filename_Si4 = 'Quartz_14deg.tif'
#exp_filename_Si5 = 'Quartz_16deg.tif'

exp_filename_Si6 = 'Sapphire_10deg.tif'
exp_filename_Si7 = 'Sapphire_14deg.tif'
exp_filename_Si8 = 'Sapphire_16deg.tif'
exp_filename_Si9 = 'Sapphire_20deg.tif'

exp_filename_Si10 = 'SiN_10deg.tif'
exp_filename_Si11 = 'SiN_15deg.tif'

#g.tifToNpzConversion(exp_filename_Si1, exp_data_directory, 'feb', 0.1)
#g.tifToNpzConversion(exp_filename_Si2, exp_data_directory, 'feb', 0.15)

#g.tifToNpzConversion(exp_filename_Si3, exp_data_directory, 'dec', 0.1)
#g.tifToNpzConversion(exp_filename_Si4, exp_data_directory, 'dec', 0.14)

#g.tifToNpzConversion(exp_filename_Si1, exp_data_directory, 'dec', 0.1)

#g.tifToNpzConversion(exp_filename_Si2, exp_data_directory, 'feb', 0.15)

#g.tifToNpzConversion(exp_filename_Si3, exp_data_directory, 'feb', 0.10)
#g.tifToNpzConversion(exp_filename_Si4, exp_data_directory, 'feb', 0.14)
#g.tifToNpzConversion(exp_filename_Si5, exp_data_directory, 'feb', 0.16)

g.tifToNpzConversion(exp_filename_Si6, exp_data_directory, 'feb', 0.10)
g.tifToNpzConversion(exp_filename_Si7, exp_data_directory, 'feb', 0.14)
g.tifToNpzConversion(exp_filename_Si8, exp_data_directory, 'feb', 0.16)
g.tifToNpzConversion(exp_filename_Si9, exp_data_directory, 'feb', 0.20)
#g.tifToNpzConversion(exp_filename_Si9, exp_data_directory, 'feb', 0.10)
#g.tifToNpzConversion(exp_filename_Si10, exp_data_directory, 'feb', 0.15)
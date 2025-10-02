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

exp_filename_Si1 = 'Mica_4824_2000RPM_3mgPml_0p35deg.tif'

g.tifToNpzConversion(exp_filename_Si1, exp_data_directory, 'feb', 0.35)
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

directory1 = r'C:\BornAgainSimulations\data\tif\feb'
filename1 = '32_15deg.tif'

g.tifToNpzConversion(filename1, directory1, 'feb', 0.15)
'''
exp_data_directory = r'C:\BornAgainSimulations\data\tif\dec'
exp_filename_Si1 = 'FAPbBr_4824_40gPL_16Precursors_10000RPM_10deg.tif'
exp_filename_Si2 = 'FAPbBr_4824_40gPL_16Precursors_10000RPM_15deg.tif'

exp_filename_Si3 = 'FAPbBr_4824_40gPL_16Precursors_1000RPM_10deg.tif'
exp_filename_Si4 = 'FAPbBr_4824_40gPL_16Precursors_1000RPM_15deg.tif'

exp_filename_Si5 = 'FAPbBr_4824_40gPL_27Precursors_10000RPM_10deg.tif'
exp_filename_Si6 = 'FAPbBr_4824_40gPL_27Precursors_10000RPM_15deg.tif'

exp_filename_Si7 = '4_16deg.tif'
exp_filename_Si8 = '4_17deg.tif'

exp_filename_Si9 = '4_18deg.tif'
exp_filename_Si10 = '4_19deg.tif'
exp_filename_Si11 = '4_20deg.tif'

#exp_filename_Si10 = 'SiN_10deg.tif'
#exp_filename_Si11 = 'SiN_15deg.tif'
g.tifToNpzConversion(exp_filename_Si1, exp_data_directory, 'dec', 0.10)
g.tifToNpzConversion(exp_filename_Si2, exp_data_directory, 'dec', 0.15)

g.tifToNpzConversion(exp_filename_Si3, exp_data_directory, 'dec', 0.10)
g.tifToNpzConversion(exp_filename_Si4, exp_data_directory, 'dec', 0.15)

g.tifToNpzConversion(exp_filename_Si5, exp_data_directory, 'dec', 0.10)
g.tifToNpzConversion(exp_filename_Si6, exp_data_directory, 'dec', 0.15)

#g.tifToNpzConversion(exp_filename_Si7, exp_data_directory, 'feb', 0.16)
#g.tifToNpzConversion(exp_filename_Si8, exp_data_directory, 'feb', 0.17)

#g.tifToNpzConversion(exp_filename_Si9, exp_data_directory, 'feb', 0.18)
#g.tifToNpzConversion(exp_filename_Si10, exp_data_directory, 'feb', 0.19)
#g.tifToNpzConversion(exp_filename_Si11, exp_data_directory, 'feb', 0.20)

#g.tifToNpzConversion(exp_filename_Si6, exp_data_directory, 'feb', 0.10)
#g.tifToNpzConversion(exp_filename_Si7, exp_data_directory, 'feb', 0.14)
#g.tifToNpzConversion(exp_filename_Si8, exp_data_directory, 'feb', 0.16)
#g.tifToNpzConversion(exp_filename_Si9, exp_data_directory, 'feb', 0.20)
#g.tifToNpzConversion(exp_filename_Si9, exp_data_directory, 'feb', 0.10)
#g.tifToNpzConversion(exp_filename_Si10, exp_data_directory, 'feb', 0.15)
'''
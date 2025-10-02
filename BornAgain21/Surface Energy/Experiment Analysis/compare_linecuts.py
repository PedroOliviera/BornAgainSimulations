from GISAXS_Analysis import GISAXS_setup_v21 as g
from GISAXS_Analysis import Graphing_Analysis as graphing

exp_data_directory = r'C:\BornAgainSimulations\data\exp-npz'
    
exp_filenames = ['Mica_0p2deg.npz','Quartz_0p2deg.npz','Sapphire_0p2deg.npz','SiN_0p2deg.npz', 'Si_0p2deg.npz']
exp_2d_array = []
exp_axes_array = []
labels = ['Mica', 'Quartz', ' Sapphire', 'SiN', 'Si']
'''
for fname in exp_filenames:
    exp_2d, exp_axes = g.load_npz_data(fname, exp_data_directory)
    data_npArrays.append(exp_2d)
'''

linecuts1 = [0.44, 0.3, 0.31, 0.275, 0.31] #0.275 - 0.33 - 0.822 SiN 0.44 Mica
linecuts2 = [0.093, 0.091, 0.089, 0.095, 0.0829]

for linecut1, linecut2, label, fname in zip(linecuts1, linecuts2, labels, exp_filenames):
    exp_2d, exp_axes = g.load_npz_data(fname, exp_data_directory)
    exp_2d_array.append(exp_2d)
    exp_axes_array.append(exp_axes)
    graphing.plot2D(exp_2d, realDat_axes=exp_axes, L1_qz=linecut1, L2_qy=linecut2, zlim=[22,exp_2d.max()])
    graphing.plt.title(label)
    graphing.linecutsItoV(experimental_data=exp_2d, L1_qz=linecut1, L2_qy=linecut2, axes_exp=exp_axes, save=True, savefname = label)
    graphing.plt.title(label)


graphing.hor_slice_comparison(hor_slice_q_array=linecuts1, 
                              data_npArrays=exp_2d_array, 
                              data_axes_array=exp_axes_array, 
                              xmin=0.06, xmax=0.125, labels=labels)
graphing.plt.xlim(right=0.6)
graphing.plt.ylim(bottom=0.000343)

#graphing.vert_slice_comparison(vert_slice_q_array=linecuts2, 
#                               data_npArrays=exp_2d_array,
#                               data_axes_array=exp_axes_array, 
#                               xmin=0.1, xmax=0.6, labels=labels)
graphing.plt.show()
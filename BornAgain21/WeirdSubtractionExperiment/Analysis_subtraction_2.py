from GISAXS_Analysis import GISAXS_setup_v21 as g
from GISAXS_Analysis import Graphing_Analysis as graphing

exp_data_directory = r'C:\BornAgainSimulations\data\exp-npz'
    
exp_filenames = ['FAPbBr3_27Pre_10000RPM_dec_10deg.npz','FAPbBr3_27Pre_10000RPM_dec_14deg.npz']
exp_2d_array = []
exp_axes_array = []
labels = ['Polymer', 'FAPbBr3']
'''
for fname in exp_filenames:
    exp_2d, exp_axes = g.load_npz_data(fname, exp_data_directory)
    data_npArrays.append(exp_2d)
'''

linecuts1 = [0.26, 0.26] #0.275 - 0.33 - 0.822 SiN 0.44 Mica
#linecuts1 = [0.285, 0.285]
linecuts2 = [0.093, 0.091]

for linecut1, linecut2, label, fname in zip(linecuts1, linecuts2, labels, exp_filenames):
    exp_2d, exp_axes = g.load_npz_data(fname, exp_data_directory)
    exp_2d_array.append(exp_2d)
    exp_axes_array.append(exp_axes)
    graphing.plot2D(exp_2d, realDat_axes=exp_axes, L1_qz=linecut1, L2_qy=linecut2, zlim=[22,exp_2d.max()])
    graphing.plt.title(label)
    graphing.linecutsItoV(experimental_data=exp_2d, L1_qz=linecut1, L2_qy=linecut2, axes_exp=exp_axes, save=True, savefname = label)
    graphing.plt.title(label)

Polymer_2d_normalized = graphing.normalize2d_by_max(exp_2d_array[0])
Perovksite_2d_normalized = graphing.normalize2d_by_max(exp_2d_array[1])



Subtracted_2d = abs(Polymer_2d_normalized - Perovksite_2d_normalized)
merged_2d, _ = graphing.stitch_detector_halves(Polymer_2d_normalized, Perovksite_2d_normalized, zero_from="right")
print(merged_2d.min())
print(merged_2d.max())
graphing.plot2D(merged_2d, realDat_axes=exp_axes, L1_qz=linecut1, L2_qy=linecut2, zlim=[0.0006,merged_2d.max()])
graphing.plot2D(Subtracted_2d, realDat_axes=exp_axes, L1_qz=linecut1, L2_qy=linecut2, zlim=[0.0006,merged_2d.max()])
graphing.plt.show()

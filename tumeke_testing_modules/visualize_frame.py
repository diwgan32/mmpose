from matplotlib.pyplot import figure
import matplotlib.image as mpimg 
from scipy import ndimage
import seaborn as sns
import matplotlib.pyplot as plt
import re
import matplotlib.patches as mpatches

def visualize_overlap(gt_df, hrnet_df, wrnch_df, frame_number):
    '''Visualizes a given frame with the ground truth, hrnet, and wrnch estimates overlayed on top'''
    
    sns.set(rc={'figure.figsize':(21,15)})
    
    gt_f = gt_df.loc[[frame_number]]
    h_f = hrnet_df.loc[[frame_number]]
    w_f = wrnch_df.loc[[frame_number]]
    
    frame_img_name = re.search(r'(\d+[.][jpg]+)', gt_f['img'].values[0]).group(0)
    file_img_name = re.findall(r'/([0-9a-zA-Z-_]+)', gt_df[0:1].img.values[0])[1]
    img = mpimg.imread('work_dirs/tumeke_testing/ground_truth_images/{}/{}'.format(file_img_name, frame_img_name))
    
    for j in range(17):
        x = "j{}_x".format(j)
        y = "j{}_y".format(j)
        l = "j{}_l".format(j)
        # Ground Truth - BLUE
        p1 = sns.scatterplot(data=gt_f, x=x, y=y, color='blue')
        p1.text(gt_f[x]-70, gt_f[y], 
             gt_f[l].values[0], horizontalalignment='left', 
             size=7, color='blue', weight='regular')
        # Hrnet Model - RED
        p2 = sns.scatterplot(data=h_f, x=x, y=y, color='red')
        p2.text(h_f[x]+20, h_f[y]-1, 
             h_f[l].values[0], horizontalalignment='left', 
             size=7, color='red', weight='regular')
        
        # Wrnch - GREEN
        p3 = sns.scatterplot(data=w_f, x=x, y=y, color='green')
        p3.text(w_f[x]+20, w_f[y]+1, 
             w_f[l].values[0], horizontalalignment='left', 
             size=7, color='green', weight='regular')
        
    blue_patch = mpatches.Patch(color='blue', label='Ground Truth')
    red_patch = mpatches.Patch(color='red', label='HRnet Model')
    green_patch = mpatches.Patch(color='green', label='Wrnch Model')
    plt.legend(handles=[blue_patch, red_patch, green_patch])
        
    plt.imshow(img)
    plt.show()
    

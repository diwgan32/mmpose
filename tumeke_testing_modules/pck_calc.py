''' PCK - Percentage of Correct Key-points

Description: https://github.com/cbsudux/Human-Pose-Estimation-101#percentage-of-correct-parts---pcp

Percentage of Correct Key-points - PCK:

- Detected joint is considered correct if the distance between the predicted and the true joint is within a certain threshold (threshold varies)
- PCKh@0.5 is when the threshold = 50% of the head bone link
- PCK@0.2 == Distance between predicted and true joint < 0.2 * torso diameter
- Sometimes 150 mm is taken as the threshold
- Head, shoulder, Elbow, Wrist, Hip, Knee, Ankle → Keypoints
- PCK is used for 2D and 3D (PCK3D)
- Higher the better

# 1) Iterate through keypoints and check 
#    - distance between predicted and true joint < 0.2 * torso diameter

# 2) for row, get jX_d (joint detected)

# 3) Get percent of all values detected (across all keypoints)
'''
import numpy as np
import pandas as pd
import seaborn as sns
from tumeke_testing_modules.load_gt_data import joint_order
from tumeke_testing_modules.load_model_data import euclidean_dist


def frmt_perc(number):
    return "{:.2%}".format(number)
    
def extract_j_pos(f, v, joint_name):
    return f['j{}_{}'.format(joint_order.index(joint_name), v)]

def get_torso_diameter(x):
    return euclidean_dist(np.array([x['pelvis_x'], x['pelvis_y']]), np.array([x['thorax_x'], x['thorax_y']]))

def vect_duclidean_dist(df1, df2, cols=['x_coord','y_coord']):
    return np.linalg.norm(df1[cols].values - df2[cols].values, axis=1)

def get_ed_by_frame(gt_df, df):
    '''Calculate distance from ground truth for each joint in a frame'''
    ed_dist_df = gt_df[[]] # Return just index
    for j in range(17):
        x = "j{}_x".format(j)
        y = "j{}_y".format(j)
        ed_dist_df['{}_ed'.format(joint_order[j])] = vect_duclidean_dist(gt_df, df.iloc[gt_df.index], cols=[x,y])
    
    return ed_dist_df
        
    
def get_pck_stats(gt_df, df, model_name):
    ''''For a given data frame, compare with ground truth using PCK'''
    
    print('-' * 10)
    print(model_name)
    print('-' * 10)
    
    # Get torso diameter for each frame (both df and gt_df)
    for df_i in [df, gt_df]:

        df_i['pelvis_x'] = (extract_j_pos(df_i, 'x', 'L_Hip') + extract_j_pos(df_i, 'x', 'R_Hip')) / 2
        df_i['pelvis_y'] = (extract_j_pos(df_i, 'y', 'L_Hip') + extract_j_pos(df_i, 'y', 'R_Hip')) / 2
        df_i['thorax_x'] = (extract_j_pos(df_i, 'x', 'L_Shoulder') + extract_j_pos(df_i, 'x', 'R_Shoulder')) / 2
        df_i['thorax_y'] = (extract_j_pos(df_i, 'y', 'L_Shoulder') + extract_j_pos(df_i, 'y', 'R_Shoulder')) / 2

        df_i['torso_diameter'] = df_i.apply(lambda x: get_torso_diameter(x), axis=1)
    
    # Set threshold
    pck_threshold_s = .20 * gt_df['torso_diameter']
    
    # Get Euclidean Distance
    ed_dist_df = get_ed_by_frame(gt_df, df)

    # Determine if joint is meeting pck threshold
    pck_df_scores = ed_dist_df.lt(pck_threshold_s, axis='index') # True if distance less than PCK threshold
    pck_df_scores = pck_df_scores.applymap(lambda x: 1 if x == True else 0)

    # % Detected frames
    print('{} | % Detected frames:'.format(model_name),
        frmt_perc(pck_df_scores.sum(axis=1).apply(lambda x: 1 if x > 0 else 0).sum() / pck_df_scores.sum(axis=1).count()))

    # Total (all frames)
    print('{} | PCK - Total (all frames):'.format(model_name),
        frmt_perc(pck_df_scores.sum(axis=1).sum() / pck_df_scores.count(axis=1).sum()))

    # Total (detected frames)
    pck_detected_df = pck_df_scores.sum(axis=1).apply(lambda x: True if x > 0 else False)
    print('{} | PCK - Total (detected frames):'.format(model_name),
        frmt_perc(pck_df_scores[pck_detected_df].sum(axis=1).sum() / pck_df_scores[pck_detected_df].count(axis=1).sum()))

    # By Body part
    print('{} | PCK - By Body part (all frames):'.format(model_name))
    print(pck_df_scores.sum() / pck_df_scores.count() * 100)
    
    print('{} | PCK - By Body part (detected frames):'.format(model_name))
    print(pck_df_scores[pck_detected_df].sum() / pck_df_scores[pck_detected_df].count() * 100)

    # by frame
    # pd.DataFrame(pck_df_scores.sum(axis=1) / pck_df_scores.count(axis=1), columns=['PCK']).plot.area()
    print('{} | PCK - By Frame:'.format(model_name))
    pd.DataFrame(pck_df_scores).plot.area(title='{} | PCK over time by joint'.format(model_name))

def pck_calc(gt_df, hrnet_df, wrnch_df):
    '''Calculate PCK stats for hrnet and wrnch'''
    
    print(
        '''
PCK - Percentage of Correct Key-points

Description: https://github.com/cbsudux/Human-Pose-Estimation-101#percentage-of-correct-parts---pcp

Percentage of Correct Key-points - PCK:

- Detected joint is considered correct if the distance between the predicted and 
      the true joint is within a certain threshold (threshold varies)
- PCKh@0.5 is when the threshold = 50% of the head bone link
- PCK@0.2 == Distance between predicted and true joint < 0.2 * torso diameter
- Sometimes 150 mm is taken as the threshold
- Head, shoulder, Elbow, Wrist, Hip, Knee, Ankle → Keypoints
- PCK is used for 2D and 3D (PCK3D)
- Higher the better
        '''
    )
    
    sns.set(rc={'figure.figsize':(21,7)})
    
    get_pck_stats(gt_df, hrnet_df, 'hrnet')
    print('-' * 20)
    get_pck_stats(gt_df, wrnch_df, 'wrnch')
    
    
    


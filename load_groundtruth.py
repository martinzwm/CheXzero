import pickle
import numpy as np
import pandas as pd
import os
if __name__ == '__main__':
    #load pickle file with ground truth labels
    with open('/home/ec2-user/CHEXLOCALIZE/CheXlocalize/gradcam_maps_val/patient64727_study1_view1_frontal_Atelectasis_map.pkl', 'rb') as f:
        ground_truth = pickle.load(f)
    print(ground_truth['cxr_img'].shape)
    print(ground_truth.keys())
    val_labels = pd.read_csv('/home/ec2-user/CHEXLOCALIZE/CheXpert/val_labels.csv')
    val_labels['cur_path'] = val_labels['Path'].apply(lambda x: os.path.join('/home/ec2-user/CHEXLOCALIZE/CheXpert/',*x.split('/')[1:]))
    print(val_labels['cur_path'][0])
    print(val_labels.head(10))
    val_labels.to_csv('/home/ec2-user/CHEXLOCALIZE/CheXpert/val_labels.csv', index=False)
import os
from utils import *

# Path to the datasets
root_path = './datasets/Market-1501'

if not os.path.exists(root_path):
    print('[!] please change the root_path')

save_path = os.path.join(root_path, 'market-reid')
if not os.path.exists(save_path):
    os.makedirs(save_path)

# -----------------------------------------
# query
# -----------------------------------------
split_subfolder(root_path,
                'query',
                save_path,
                'query')

# ----------------------------------------
# multi-query
# ----------------------------------------
split_subfolder(root_path,
                'gt_bbox',
                save_path,
                'multi-query')

# -----------------------------------------
# gallery
# -----------------------------------------
split_subfolder(root_path,
                'bounding_box_test',
                save_path,
                'gallery')

# -----------------------------------------
# train_all
# -----------------------------------------
split_subfolder(root_path,
                'bounding_box_train',
                save_path,
                'train_all')

# -----------------------------------------
# train_val
# -----------------------------------------
split_train_val(root_path,
                'bounding_box_train',
                save_path,
                'train',
                'val')

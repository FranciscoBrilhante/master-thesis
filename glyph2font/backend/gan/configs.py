def conf_default():
    return {
        'logname': 'default_seed_0',
        'device': 'cuda:0',
        'seed': 0,
        'path_log': '../logs/experiment1',
        'n_visible': 3,
        'epochs': 300,
        'epochs_decay_lr': 100,
        'batch_size': 64,
        'lr': 0.0002,
        'beta': 0.5,
        'l1_lambda': 100,
        #"glyphs_visible": [7,    14,    3,    18,    6,    17],
        'data_path': '/home/francisco/dataset/capitals64Original',
        'data_path_test': '/home/francisco/mc-gan/datasets/Capitals64/test',
        'use_mask': False,
        'use_skeleton_mask': False,
        'masks_path': '/home/francisco/dataset/masks/train/0',
        'default_font': '/home/francisco/mc-gan/datasets/Capitals64/BASE/Code New Roman.0.0.png'
    }

def conf_experiment1_var1():
    return {
        'logname': 'default_seed_0',
        'device': 'cuda:0',
        'seed': 0,
        'path_log': '../logs/experiment1',
        'n_visible': 3,
        'epochs': 300,
        'epochs_decay_lr': 100,
        'batch_size': 64,
        'lr': 0.0002,
        'beta': 0.5,
        'l1_lambda': 100,
        #"glyphs_visible": [7,    14,    3,    18,    6,    17],
        'data_path': '/home/francisco/dataset/capitals64Original',
        'data_path_test': '/home/francisco/mc-gan/datasets/Capitals64/test',
        'use_mask': False,
        'use_skeleton_mask': False,
        'masks_path': '/home/francisco/dataset/masks/train/0',
        'default_font': '/home/francisco/mc-gan/datasets/Capitals64/BASE/Code New Roman.0.0.png'
    }

def conf_experiment1_var2():
    return {
        'logname': 'mask_seed_0',
        'device': 'cuda:0',
        'seed': 0,
        'path_log': '../logs/experiment1',
        'n_visible': 3,
        'epochs': 300,
        'epochs_decay_lr': 100,
        'batch_size': 64,
        'lr': 0.0002,
        'beta': 0.5,
        'l1_lambda': 100,
        #"glyphs_visible": [7,    14,    3,    18,    6,    17],
        'data_path': '/home/francisco/dataset/capitals64Original',
        'data_path_test': '/home/francisco/mc-gan/datasets/Capitals64/test',
        'use_mask': True,
        'use_skeleton_mask': False,
        'masks_path': '/home/francisco/dataset/masks/train/0',
        'default_font': '/home/francisco/mc-gan/datasets/Capitals64/BASE/Code New Roman.0.0.png'
    }

def conf_experiment1_var3():
    return {
        'logname': 'skeleton_seed_0',
        'device': 'cuda:0',
        'seed': 0,
        'path_log': '../logs/experiment1',
        'n_visible': 3,
        'epochs': 300,
        'epochs_decay_lr': 100,
        'batch_size': 64,
        'lr': 0.0002,
        'beta': 0.5,
        'l1_lambda': 100,
        #"glyphs_visible": [7,    14,    3,    18,    6,    17],
        'data_path': '/home/francisco/dataset/capitals64Original',
        'data_path_test': '/home/francisco/mc-gan/datasets/Capitals64/test',
        'use_mask': True,
        'use_skeleton_mask': True,
        'masks_path': '/home/francisco/dataset/masks/train/0',
        'default_font': '/home/francisco/mc-gan/datasets/Capitals64/BASE/Code New Roman.0.0.png'
    }

# {K,H,B,N,R,D} subset 1
# [0,2,4,5,6,8,9,11,12,14,15,16,18,19,20,21,22,23,24,25]

# {H,O,D,S,G,R} subset 2
# [7,14,3,18,6,17]

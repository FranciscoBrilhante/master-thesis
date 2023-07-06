def conf_default():
    return {
        'logname': 'default_conditional',
        'device': 'cuda:0',
        'seed': 0,
        'path_log': '../logs/diffusion',
        'epochs': 60,
        'batch_size': 24,
        'lr': 0.0001,
        'max_steps':1_000_000,
        'max_beta':0.02,
        'train_data': '/home/francisco/dataset/capitals64/train',
        'test_data': '/home/francisco/dataset/capitals64/test',
        'style_dir': '/home/francisco/dataset/capitals64Features/train',
        'style_dir_test': '/home/francisco/dataset/capitals64Features/test',
        'checkpoint_dir': '/home/francisco/logs/diffusion/default_conditional/Checkpoints/last.pt',
    }


def conf_glyphnet():
    return {
        'logname': 'glyphnet',
        'device': 'cuda:0',
        'seed': 0,
        'path_log': '../logs/diffusion',
        'n_visible': 6,
        'epochs': 600,
        'epochs_decay_lr': 100,
        'batch_size': 64,
        'lr': 0.0002,
        'beta': 0.5,
        'l1_lambda': 100,
        "glyphs_visible": [7, 14, 3, 18, 6, 17],
        'data_path': '/home/francisco/dataset/capitals64',
        'data_path_test': '/home/francisco/dataset/capitals64/test',
        'use_mask': True,
        'use_skeleton_mask': False,
        'masks_path': '/home/francisco/dataset/masks/train/0',
        'default_font': '/home/francisco/mc-gan/datasets/Capitals64/BASE/Code New Roman.0.0.png'
    }

def conf_default_2():
    return {
        'logname': 'test(delete)',
        'device': 'cuda:0',
        'seed': 0,
        'path_log': '../logs/diffusion',
        'epochs': 60,
        'batch_size': 24,
        'lr': 0.0001,
        'max_steps':1_000_000,
        'max_beta':0.02,
        'train_data': '/home/francisco/dataset/capitals64/train',
        'test_data': '/home/francisco/dataset/capitals64/test',
        'style_dir': '/home/francisco/dataset/capitals64Features/train',
        'style_dir_test': '/home/francisco/dataset/capitals64Features/test',
        'checkpoint_dir': None
    }
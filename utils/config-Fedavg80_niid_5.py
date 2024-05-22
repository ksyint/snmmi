import os

# import models.unet as unet
# anp -e 'grp=hpc proj=Fedprox64_non_iid_5 epc=1 end=20' config_run_fed.yml

Striatum_ADNI = {
    'description': 'Striatum',
    'train_dir': os.path.join('data', 'FREESURFER_ADNI_TRAIN'),
    'test_dir': os.path.join('data', 'FREESURFER_ADNI_TEST'),
    'orig_file': 'orig.nii',
    'mask_file': 'aseg.nii',
    'classes': [
        'Left-Cerebellum-Cortex', # 8
        'Left-Caudate',  # 11
        'Left-Putamen',  # 12
        'Right-Cerebellum-Cortex', # 47
        'Right-Caudate',  # 50
        'Right-Putamen',  # 51
    ],
    'lr': 1e-3,
    'scheduler_step': 1,
    'scheduler_gamma': 1.0,
    'w_bce': 0.0,
    'w_dice': 1.0,
    'w_l1': 0.0,
    'w_penalty': 0.0,
    'epsilon': 1e-5,
    'preprocess': False,
    'canonical': False,
    'deformation': 0.0,
    'biasfield': 0.2,
    'noise': 0.0,
    'flip': 0.0,
    'affine': 0.5,
    'zoom': 1.3,
    'znorm': True,

    'strategy': 'fedprox',
    'mu': 0.1,
    'resolution': 80,
    'batch_size': 5,
    'num_workers': 10,
}

VM100_1 = {
    'seed': 1,
    **Striatum_ADNI,
    'train_dir': os.path.join('data', 'FREESURFER_GAAIN_TRAIN'),
    'test_dir': os.path.join('data', 'FREESURFER_GAAIN_TEST'),
    'orig_file': 'orig.nii.gz',
    'mask_file': 'aseg.nii.gz',
    'weights': [
      2.,
      3.,
      3.,
      2.,
      3.,
      3.,
    ],
}

VM100_2 = {
    'seed': 2,
    **Striatum_ADNI,
    'train_dir': os.path.join('data', 'FREESURFER_GAAIN_TRAIN'),
    'test_dir': os.path.join('data', 'FREESURFER_GAAIN_TEST'),
    'orig_file': 'orig.nii.gz',
    'mask_file': 'aseg.nii.gz',
    'weights': [
      1.,
      2.,
      2.,
      1.,
      2.,
      2.,
    ],
}

VP100_1 = {
    'seed': 3,
    **Striatum_ADNI,
    'weights': [
      2.,
      3.,
      3.,
      2.,
      3.,
      3.,
    ],
}

VP100_2 = {
    'seed': 4,
    **Striatum_ADNI,
    'weights': [
      2.,
      3.,
      3.,
      2.,
      3.,
      3.,
    ],
}

VV100_1 = {
    'seed': 5,
    **Striatum_ADNI,
    'weights': [
      1.,
      2.,
      2.,
      1.,
      2.,
      2.,
    ],
}

VV100_2 = {
    'seed': 6,
    **Striatum_ADNI,
    'weights': [
      1.,
      2.,
      2.,
      1.,
      2.,
      2.,
    ],
}

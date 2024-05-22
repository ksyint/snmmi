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
    'zoom': 1.2,
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
    'orig_file': 'orig.nii',
    'mask_file': 'aseg.nii',
    'classes': [
        # 'Left-Cerebral-White-Matter', # 2
        # 'Left-Cerebral-Cortex', # 3
        'Left-Cerebellum-Cortex', # 8
        'Left-Caudate', # 11
        'Left-Putamen', # 12
        # 'Right-Cerebral-White-Matter', # 41
        # 'Right-Cerebral-Cortex', # 42
        'Right-Cerebellum-Cortex', # 47
        'Right-Caudate', # 50
        'Right-Putamen', # 51
    ],
    'weights': [
      1.,
      1.,
      1.,
      1.,
      1.,
      1.,
    ],
}

VM100_2 = {
    'seed': 1,
    **Striatum_ADNI,
    'train_dir': os.path.join('data', 'FREESURFER_GAAIN_TRAIN'),
    'test_dir': os.path.join('data', 'FREESURFER_GAAIN_TEST'),
    'orig_file': 'orig.nii',
    'mask_file': 'aseg.nii',
    'classes': [
        # 'Left-Cerebral-Cortex', # 3
        'Left-Cerebellum-Cortex', # 8
        'Left-Caudate', # 11
        'Left-Putamen', # 12
        # 'ctx-lh-caudalmiddlefrontal', # 1003
        # 'ctx-lh-inferiorparietal', # 1008
        # 'ctx-lh-inferiortemporal', # 1009
        # 'ctx-lh-middletemporal', # 1015
        # 'ctx-lh-postcentral', # 1022
        # 'ctx-lh-precentral', # 1024
        # 'ctx-lh-rostralmiddlefrontal', # 1027
        # 'ctx-lh-superiorfrontal', # 1028
        # 'ctx-lh-superiorparietal', # 1029
        # 'ctx-lh-superiortemporal', # 1030
        # 'ctx-lh-supramarginal', # 1031
        
        # 'Right-Cerebral-Cortex', # 42
        'Right-Cerebellum-Cortex', # 47
        'Right-Caudate', # 50
        'Right-Putamen', # 51
        # 'ctx-rh-caudalmiddlefrontal', # 2003
        # 'ctx-rh-inferiorparietal', # 2008
        # 'ctx-rh-inferiortemporal', # 2009
        # 'ctx-rh-middletemporal', # 2015
        # 'ctx-rh-postcentral', # 2022
        # 'ctx-rh-precentral', # 2024
        # 'ctx-rh-rostralmiddlefrontal', # 2027
        # 'ctx-rh-superiorfrontal', # 2028
        # 'ctx-rh-superiorparietal', # 2029
        # 'ctx-rh-superiortemporal', # 2030
        # 'ctx-rh-supramarginal', # 2031
    ],
    'weights': [
      3.,
      2.,
      2.,
      2.,
      1.,
      1.,
    ],
}

VP100_1 = {
    'seed': 2,
    **Striatum_ADNI,
    'classes': [
        # 'Left-Cerebral-White-Matter', # 2
        # 'Left-Cerebral-Cortex', # 3
        'Left-Cerebellum-Cortex', # 8
        'Left-Caudate', # 11
        'Left-Putamen', # 12
        # 'Right-Cerebral-White-Matter', # 41
        # 'Right-Cerebral-Cortex', # 42
        'Right-Cerebellum-Cortex', # 47
        'Right-Caudate', # 50
        'Right-Putamen', # 51
    ],
    'weights': [
      1.,
      1.,
      1.,
      1.,
      1.,
      1.,
    ],
}

VP100_2 = {
    'seed': 3,
    **Striatum_ADNI,
    'classes': [
        # 'Left-Cerebral-White-Matter', # 2
        # 'Left-Cerebral-Cortex', # 3
        'Left-Cerebellum-Cortex', # 8
        'Left-Caudate', # 11
        'Left-Putamen', # 12
        # 'Right-Cerebral-White-Matter', # 41
        # 'Right-Cerebral-Cortex', # 42
        'Right-Cerebellum-Cortex', # 47
        'Right-Caudate', # 50
        'Right-Putamen', # 51
    ],
    'weights': [
      1.,
      1.,
      1.,
      1.,
      1.,
      1.,
    ],
}

VV100_1 = {
    'seed': 2,
    **Striatum_ADNI,
    'orig_file': 'orig.nii',
    'mask_file': 'aseg.nii',
    'classes': [
        # 'Left-Cerebral-Cortex', # 3
        'Left-Cerebellum-Cortex', # 8
        'Left-Caudate', # 11
        'Left-Putamen', # 12
        # 'ctx-lh-caudalmiddlefrontal', # 1003
        # 'ctx-lh-inferiorparietal', # 1008
        # 'ctx-lh-inferiortemporal', # 1009
        # 'ctx-lh-middletemporal', # 1015
        # 'ctx-lh-postcentral', # 1022
        # 'ctx-lh-precentral', # 1024
        # 'ctx-lh-rostralmiddlefrontal', # 1027
        # 'ctx-lh-superiorfrontal', # 1028
        # 'ctx-lh-superiorparietal', # 1029
        # 'ctx-lh-superiortemporal', # 1030
        # 'ctx-lh-supramarginal', # 1031
        # 'Right-Cerebral-Cortex', # 42
        'Right-Cerebellum-Cortex', # 47
        'Right-Caudate', # 50
        'Right-Putamen', # 51
        # 'ctx-rh-caudalmiddlefrontal', # 2003
        # 'ctx-rh-inferiorparietal', # 2008
        # 'ctx-rh-inferiortemporal', # 2009
        # 'ctx-rh-middletemporal', # 2015
        # 'ctx-rh-postcentral', # 2022
        # 'ctx-rh-precentral', # 2024
        # 'ctx-rh-rostralmiddlefrontal', # 2027
        # 'ctx-rh-superiorfrontal', # 2028
        # 'ctx-rh-superiorparietal', # 2029
        # 'ctx-rh-superiortemporal', # 2030
        # 'ctx-rh-supramarginal', # 2031
    ],
    'weights': [
      3.,
      1.,
      2.,
      1.,
      1.,
      1.,
    ],
}

VV100_2 = {
    'seed': 3,
    **Striatum_ADNI,
    'orig_file': 'orig.nii',
    'mask_file': 'aseg.nii',
    'classes': [
        # 'Left-Cerebral-Cortex', # 3
        'Left-Cerebellum-Cortex', # 8
        'Left-Caudate', # 11
        'Left-Putamen', # 12
        # 'ctx-lh-caudalmiddlefrontal', # 1003
        # 'ctx-lh-inferiorparietal', # 1008
        # 'ctx-lh-inferiortemporal', # 1009
        # 'ctx-lh-middletemporal', # 1015
        # 'ctx-lh-postcentral', # 1022
        # 'ctx-lh-precentral', # 1024
        # 'ctx-lh-rostralmiddlefrontal', # 1027
        # 'ctx-lh-superiorfrontal', # 1028
        # 'ctx-lh-superiorparietal', # 1029
        # 'ctx-lh-superiortemporal', # 1030
        # 'ctx-lh-supramarginal', # 1031
        # 'Right-Cerebral-Cortex', # 42
        'Right-Cerebellum-Cortex', # 47
        'Right-Caudate', # 50
        'Right-Putamen', # 51
        # 'ctx-rh-caudalmiddlefrontal', # 2003
        # 'ctx-rh-inferiorparietal', # 2008
        # 'ctx-rh-inferiortemporal', # 2009
        # 'ctx-rh-middletemporal', # 2015
        # 'ctx-rh-postcentral', # 2022
        # 'ctx-rh-precentral', # 2024
        # 'ctx-rh-rostralmiddlefrontal', # 2027
        # 'ctx-rh-superiorfrontal', # 2028
        # 'ctx-rh-superiorparietal', # 2029
        # 'ctx-rh-superiortemporal', # 2030
        # 'ctx-rh-supramarginal', # 2031
    ],
    'weights': [
      3.,
      1.,
      2.,
      1.,
      1.,
      1.,
    ],
}

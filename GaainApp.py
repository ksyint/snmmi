import argparse
import datetime
import os
import shutil
import copy
import sys
import signal
import atexit  # 이 부분을 추가합니다.
import numpy as np
import json
import torch
import torch.nn as nn
import torchio as tio
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import random
import models.unet as unet
import utils.logconf as logconf
from dsets.GaainDataset import GaainDataset
from utils.logconf import logging
from utils.tools import *
from utils.config import * 
from utils.tools import enumerateWithEstimate


# 현재 프로세스의 PID를 main_pid로 설정
main_pid = os.getpid()

# main_pid = os.getpid()
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.INFO)



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    
    # # worker_seed = seed + worker_id
    # worker_seed = base_seed * 1000 + worker_id
    # np.random.seed(worker_seed)
    # random.seed(worker_seed)
    # torch.manual_seed(worker_seed)


def create_histogram_landmarks(reference_image_paths, output_path='utils/landmarks.json'):
    landmarks = tio.HistogramStandardization.train(reference_image_paths)
    
    # NumPy 배열을 Python 리스트로 변환
    landmarks_list = landmarks.tolist()

    # JSON 파일로 저장
    with open(output_path, 'w') as f:
        json.dump(landmarks_list, f)
    return landmarks

def fix_seed(random_seed, use_cuda):
    """
    fix seed to control any randomness from a code 
    (enable stability of the experiments' results.)
    """
    torch.manual_seed(random_seed)
    if use_cuda:
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

class GaainApp:
    NORM_COLORDICT = {k: [v_i / 255.0 for v_i in v] for k, v in COLORDICT.items()}
    
    # basic_dict = BASIC_DICT
    # small_dict = SMALL_DICT
    # large_dict = LARGE_DICT
    # mixed_dict = MIXED_DICT
    # cerebral_cortex_dict = BOTH_CBLCORTEX_DICT
    # cerebral_wm_dict = BOTH_CBLWM_DICT
    # cerebellum_cortex_dict = BOTH_CBMCORTEX_DICT
    # cerebellum_wm_dict = BOTH_CBMWM_DICT
    
    def __init__(self, args):
        self.selected_gpu_count = 1
        self.cli_args = args
        self.time_str = ctime
        # self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.totalTrainingSamples_count = 0

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            if self.cli_args.gpus is not None:
                gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpus.split(',')]
                self.device = torch.device(f"cuda:{gpu_ids[0]}")  # 첫 번째 GPU ID를 사용
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        analaysis_list = [
            VM100_1,
            VM100_2,
            VP100_1,
            VP100_2,
            VV100_1,
            VV100_2,
        ]
        
        if self.cli_args.analysis < len(analaysis_list):
            self.analysis_dict = analaysis_list[self.cli_args.analysis]
        else:
            self.analysis_dict = analaysis_list[0]

        # landmark_path = 'utils/landmarks.json'
        # if os.path.exists(landmark_path):
        #     # 히스토그램 랜드마크 JSON 파일 불러오기
        #     with open(landmark_path, 'r') as f:
        #         landmarks_list = json.load(f)
        # else:
        #     src_path = self.analysis_dict['train_dir']
        #     reference_image_paths = [os.path.join(src_path, el, 'orig.nii.gz') for el in os.listdir(src_path)][:5]
        #     landmarks_list = create_histogram_landmarks(reference_image_paths, )
        # # Python 리스트를 NumPy 배열로 변환
        # self.histogram_landmarks = {'MRI': np.array(landmarks_list)}

        self.cli_args.num_workers = self.analysis_dict['num_workers']
        self.cli_args.batch_size = self.analysis_dict['batch_size']
        self.cli_args.seed = self.analysis_dict['seed']
        self.cli_args.lr = self.analysis_dict['lr']
        self.cli_args.scheduler_step = self.analysis_dict['scheduler_step']
        self.cli_args.scheduler_gamma = self.analysis_dict['scheduler_gamma']
        self.cli_args.w_bce = self.analysis_dict['w_bce']
        self.cli_args.w_dice = self.analysis_dict['w_dice']
        self.cli_args.w_l1 = self.analysis_dict['w_l1']
        self.cli_args.w_penalty = self.analysis_dict['w_penalty']
        self.cli_args.epsilon = self.analysis_dict['epsilon']
        self.cli_args.preprocess = self.analysis_dict['preprocess']
        self.cli_args.resolution = self.analysis_dict['resolution']
        self.cli_args.strategy = self.analysis_dict['strategy']
        self.cli_args.canonical = self.analysis_dict['canonical']
        self.cli_args.deformation = self.analysis_dict['deformation']
        self.cli_args.biasfield = self.analysis_dict['biasfield']
        self.cli_args.noise = self.analysis_dict['noise']
        self.cli_args.flip = self.analysis_dict['flip']
        self.cli_args.affine = self.analysis_dict['affine']
        self.cli_args.zoom = self.analysis_dict['zoom']
        self.cli_args.znorm = self.analysis_dict['znorm']
        
        # 전역 난수 생성기 초기화
        self.generator = torch.Generator()
        self.generator.manual_seed(self.cli_args.seed)

        # 난수 시드 고정
        fix_seed(self.cli_args.seed, self.use_cuda)
        # torch.manual_seed(self.cli_args.seed)
        # if self.use_cuda:
        #     torch.cuda.manual_seed(self.cli_args.seed)
        #     torch.cuda.manual_seed_all(self.cli_args.seed)
        # np.random.seed(self.cli_args.seed)
        # random.seed(self.cli_args.seed)

        self.ref_t_writer = None
        self.ref_v_writer = None
        self.trn_writer = None
        self.val_writer = None
        self.initTensorboardWriters()

        self.train_dir = self.analysis_dict['train_dir']
        self.test_dir = self.analysis_dict['test_dir']
        self.orig_file = self.analysis_dict['orig_file']
        self.mask_file = self.analysis_dict['mask_file']
        self.classes = self.analysis_dict['classes']
        self.num_classes = self.analysis_dict['classes'].__len__()
        self.weights = self.analysis_dict['weights']
        self.mu = self.analysis_dict['mu']
        self.bce_loss = torch.nn.BCELoss()
        self.l1_loss = torch.nn.L1Loss()

        flipped = {value: key for key, value in CLASSES.items()}
        self.index_classes = sorted([flipped[el] for el in self.classes])

        self.preprocessing_list = ['skullstriping'] if self.cli_args.preprocess else None
            
        self.transforms_dict = {
            **(
                {'zoom': tio.RandomAffine(
                    scales=(1.0, self.cli_args.zoom),
                    degrees=0,
                    translation=(0, 0, 0),
                    image_interpolation='bspline',  # 'bspline' 이미지에 적용할 보간 방법
                    p=1.0,
                )} if self.cli_args.zoom > 1 else {}
            ),
            **(
                {'resize': tio.Resize(
                        self.cli_args.resolution
                )} if True else {}
            ),
            **(
                {'canonical': tio.ToCanonical(),
                } if self.cli_args.canonical else {}
            ),
            **(
                {'deformation': tio.RandomElasticDeformation(
                    num_control_points=(5, 5, 5),  # 제어점의 수를 정의
                    max_displacement=(3, 3, 3),  # 최대 변위량을 정의
                    p=self.cli_args.deformation,
                )} if self.cli_args.deformation > 0 else {}
            ),
            **(
                {'biasfield': tio.RandomBiasField(
                    p=self.cli_args.biasfield
                )} if self.cli_args.biasfield > 0 else {}
            ),
            **(
                {'noise': tio.RandomNoise(
                    p=self.cli_args.noise
                )} if self.cli_args.noise > 0 else {}
            ),
            **(
                {'flip': tio.RandomFlip(
                    axes=(0, 1, 2), 
                    p=self.cli_args.flip
                )} if self.cli_args.flip > 0 else {}
            ),
            **(
                {'affine': tio.RandomAffine(
                    degrees=(3, 3, 3),
                    translation=(0, 0, 0),
                    image_interpolation='bspline',  # 이미지에 적용할 보간 방법
                    p=self.cli_args.affine,
                )} if self.cli_args.affine > 0 else {}
            ),
            **(
                {'znorm': tio.ZNormalization()} if self.cli_args.znorm else {}
            ),
        }

        self.transforms_dict_test = {
            **(
                {'zoom': tio.RandomAffine(
                    scales=(self.cli_args.zoom, self.cli_args.zoom),
                    image_interpolation='bspline',  # 'bspline' 이미지에 적용할 보간 방법
                    p=1.0,
                )} if self.cli_args.zoom > 1 else {}
            ),
            **(
                {'resize': tio.Resize(
                        self.cli_args.resolution
                )} if True else {}
            ),
            **(
                {'canonical': tio.ToCanonical(),
                } if self.cli_args.canonical else {}
            ),
            **(
                {'znorm': tio.ZNormalization()} if self.cli_args.znorm else {}
            ),
        }
        
        self.segmentation_model = self.initModel()
        if self.cli_args.strategy == 'fedprox':
            self.segmentation_model_global = copy.deepcopy(self.segmentation_model)
            log.info(f"Using Strategy; {self.cli_args.strategy}:{self.mu} (fedprox:mu).")
        else:
            self.segmentation_model_global = None
            log.info(f"Using Strategy; {self.cli_args.strategy} (fedavg).")
        self.optimizer = self.initOptimizer()
        self.scheduler = self.initScheduler(self.optimizer,
                                            self.cli_args.scheduler_step,
                                            self.cli_args.scheduler_gamma
                                            )

    def initTrainDl(self):
        train_ds = GaainDataset(
            patient_dir=self.train_dir,
            images_dir=self.orig_file,
            masks_dir=self.mask_file,
            classes=self.classes,
            preprocessing=self.preprocessing_list,
            augmentation=tio.Compose([
                # tio.Resize(self.cli_args.resolution),
                # *self.train_transforms_list
                *[val for val in self.transforms_dict.values()]
            ]),
            # histogram_landmarks=self.histogram_landmarks  # 여기에 랜드마크 전달
        )
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= self.selected_gpu_count

        if os.name == 'nt':  # Windows 환경인 경우
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                num_workers=self.cli_args.num_workers,
                pin_memory=self.use_cuda,
                drop_last=True,
                shuffle=True
            )
        else:  # Windows가 아닌 다른 환경인 경우
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                num_workers=self.cli_args.num_workers,
                pin_memory=self.use_cuda,
                # persistent_workers=self.use_cuda,
                drop_last=True,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=self.generator  # 여기에 generator 추가
            )
        return train_loader

    def initValDl(self):
        test_ds = GaainDataset(
            patient_dir=self.test_dir,
            images_dir=self.orig_file,
            masks_dir=self.mask_file,
            classes=self.classes,
            preprocessing=self.preprocessing_list,
            augmentation=tio.Compose([
                # tio.Resize(self.cli_args.resolution),
                # *self.test_transforms_list
                *[val for val in self.transforms_dict_test.values()]
            ]),
            # histogram_landmarks=self.histogram_landmarks  # 여기에 랜드마크 전달
        )
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= self.selected_gpu_count

        if os.name == 'nt':  # Windows 환경인 경우
            test_loader = DataLoader(
                test_ds,
                batch_size=batch_size,
                num_workers=self.cli_args.num_workers,
                pin_memory=self.use_cuda,
                # persistent_workers=self.use_cuda,
                drop_last=True,
                shuffle=False,
            )
        else:  # Windows가 아닌 다른 환경인 경우
            test_loader = DataLoader(
                test_ds,
                batch_size=batch_size,
                num_workers=self.cli_args.num_workers,
                pin_memory=self.use_cuda,
                # persistent_workers=self.use_cuda,
                drop_last=True,
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=self.generator  # 여기에 generator 추가
            )
        return test_loader

    def initScheduler(self, optimizer, step_size=1, gamma=1):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=gamma)
    def initOptimizer(self):
        return torch.optim.Adam(self.segmentation_model.parameters(), lr=self.cli_args.lr)
        # return SGD(self.segmentation_model.parameters(), lr=0.001, momentum=0.99)

    def calculate_proximal_term(self, local_model_parameters, global_model_parameters):
        proximal_term = 0.0
        for local_param, global_param in zip(local_model_parameters, global_model_parameters):
            proximal_term += ((local_param - global_param) ** 2).sum()
        # proximal_term *= mu
        return proximal_term

    def multi_label_dice_loss(self, predicted, targets, num_classes, weights=None):
        """Calculate Dice loss for multiple labels."""
        # weights = [1.] * num_classes
        if weights is None:
            # 최소한 하나의 '1'을 포함하기 위한 조건을 만족할 때까지 반복
            while True:
                # [0, 1] 사이의 랜덤한 값을 가진 크기가 4인 텐서 생성
                weights = torch.randint(0, 2, (num_classes,))
                # 만약 텐서에 최소한 하나의 '1'이 있다면 반복 중지
                if weights.sum() >= 1:
                    weights.tolist()
                    break

        total_loss = 0.
        labels_losses = torch.zeros(num_classes, targets.shape[0], device=self.device)
        for i in range(num_classes):
            dice_loss_for_class = self.dice_loss(predicted[:, i], targets[:, i], epsilon=self.cli_args.epsilon)
            labels_losses[i, :] = dice_loss_for_class
            total_loss += weights[i] * dice_loss_for_class
        return total_loss / sum(weights), labels_losses

    def multi_label_bce_loss(self, predicted, targets, num_classes):
        # N, C, D, H, W = predicted.shape
        total_loss = 0.0
        for i in range(num_classes):  # 각 클래스별로 BCELoss 계산
            prediction_slice = predicted[:, i]
            target_slice = targets[:, i]
            loss = self.bce_loss(prediction_slice, target_slice)
            total_loss += loss
        # return total_loss / C  # 클래스 갯수로 나눠 평균 반환
        return total_loss / num_classes  # 클래스 갯수로 나눠 평균 반환

    def multi_label_l1_loss(self, predicted, targets, num_classes):
        # N, C, D, H, W = predicted.shape
        total_loss = 0.0
        for i in range(num_classes):  # 각 클래스별로 BCELoss 계산
            prediction_slice = predicted[:, i]
            target_slice = targets[:, i]
            loss = self.l1_loss(prediction_slice, target_slice)
            total_loss += loss
        # return total_loss / C  # 클래스 갯수로 나눠 평균 반환
        return total_loss / num_classes  # 클래스 갯수로 나눠 평균 반환

    def dice_loss(self, prediction_g, label_g, epsilon=1):
        # prediction_g = torch.sigmoid(prediction_g)
        diceLabel_g = label_g.sum(dim=tuple(range(1, label_g.dim())))
        dicePrediction_g = prediction_g.sum(dim=tuple(range(1, prediction_g.dim())))
        diceCorrect_g = (prediction_g * label_g).sum(dim=tuple(range(1, label_g.dim())))

        # diceRatio_g = (2 * diceCorrect_g + epsilon) / (dicePrediction_g + diceLabel_g + epsilon)
        diceRatio_g = (2 * diceCorrect_g) / (dicePrediction_g + diceLabel_g + epsilon)

        return 1 - diceRatio_g
    
    def initWeights(self):
        if self.cli_args.counter == 0:
            self.cli_args.counter = 1
        elif self.cli_args.counter == 1:
            path = os.path.join('models', self.cli_args.group, self.cli_args.project, 'init_model.state')
            init_unet_dict = torch.load(path)
        else:
            path = os.path.join('models', self.cli_args.group, self.cli_args.project, f'{self.cli_args.group}_avg_{self.cli_args.counter}.state')
            init_unet_dict = torch.load(path)
            # 모델의 가중치 로드
            segmentation_model.load_state_dict(init_unet_dict['losses'])
            # log.info(f"Valid Losses at {type(self).__name__}: {self.analysis_dict['description']}, {self.cli_args}")
            

    def initModel(self):
        segmentation_model = unet.UNet3D(
            in_channels=1,
            # num_classes=dset.class_values.__len__(),
            num_classes=self.num_classes,
        )

        if self.cli_args.counter == 0:
            self.cli_args.counter = 1
        elif self.cli_args.counter == 1:
            # path = self.analysis_dict['model'][self.cli_args.counter]
            path = os.path.join('models', self.cli_args.group, self.cli_args.project, 'init_model.state')
            init_unet_dict = torch.load(path)
            # 모델의 가중치 로드
            segmentation_model.load_state_dict(init_unet_dict['model_state'])
        else:
            # path = self.analysis_dict['model'][self.cli_args.counter]
            path = os.path.join('models', self.cli_args.group, self.cli_args.project, f'{self.cli_args.group}_avg_{self.cli_args.counter}.state')
            init_unet_dict = torch.load(path)
            # 모델의 가중치 로드
            segmentation_model.load_state_dict(init_unet_dict['model_state'])

        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                if self.cli_args.gpus is not None:
                    gpu_ids = [int(gpu_id.strip()) for gpu_id in self.cli_args.gpus.split(',')]
                    self.selected_gpu_count = len(gpu_ids)
                    segmentation_model = nn.DataParallel(segmentation_model, device_ids=gpu_ids)
                else:
                    segmentation_model = nn.DataParallel(segmentation_model)
                    self.selected_gpu_count = torch.cuda.device_count()
                segmentation_model = segmentation_model.to(self.device)
            else:
                segmentation_model = segmentation_model.to(self.device)
                self.selected_gpu_count = 1  # 기본적으로 하나의 GPU를 가정합니다.
            log.info(f"Using CUDA; {self.selected_gpu_count} devices.")
        return segmentation_model

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            valMetrics_g = torch.zeros(OFFSET_METRIC + self.num_classes, len(val_dl.dataset), device=self.device)
            self.segmentation_model.eval()

            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                loss_var, latent_vectors, labels = self.computeBatchLoss(
                    batch_ndx,
                    batch_tup,
                    val_dl.batch_size,
                    valMetrics_g,
                    random_weight=False,
                )

                # # latent_vectors가 3차원 텐서라고 가정했을 때 이를 2차원으로 변환
                # latent_variables_2d = latent_vectors.view(val_dl.batch_size, -1) 
                # integer_labels = torch.argmax(labels, dim=1).cpu().numpy()
                # # 이제 latent_variables_reduced를 TensorBoard에 기록합니다.
                # self.tsne_writer.add_embedding(
                #     mat=latent_variables_2d,
                #     metadata=integer_labels,  # metadata로 label 정보를 사용
                #     global_step=batch_ndx,
                #     tag='Latent_PCA'
                # )

        return valMetrics_g.to('cpu')

    def doTraining(self, epoch_ndx, train_dl):
        trnMetrics_g = torch.zeros(OFFSET_METRIC + self.num_classes, len(train_dl.dataset), device=self.device)
        self.segmentation_model.train()

        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var, _, _ = self.computeBatchLoss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trnMetrics_g,
                random_weight=False,
            )
            loss_var.backward()

            self.optimizer.step()

        self.totalTrainingSamples_count += trnMetrics_g.size(1)

        return trnMetrics_g.to('cpu')


    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size,
                         metrics_g,
                         random_weight=False,
                         classificationThreshold=0.5,
                         ):
        w_bce, w_dice, w_l1, w_penalty = self.cli_args.w_bce, self.cli_args.w_dice, self.cli_args.w_l1, self.cli_args.w_penalty
        
        input_t, label_t, labels_t = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        labels_g = labels_t.to(self.device, non_blocking=True)

        # if self.segmentation_model.training and self.augmentation_dict:
        #     input_g, label_g = self.augmentation_model(input_g, label_g)

        prediction_g, latent_variables = self.segmentation_model(input_g)

        prediction_g = prediction_g.unsqueeze(2)
        prediction_g = torch.sigmoid(prediction_g)
        bceLoss_g = self.multi_label_bce_loss(prediction_g, labels_g, self.num_classes) if w_bce > 0. else 0
        
        if random_weight:
            weights = None
        else:
            weights = self.weights
        
        diceLoss_g, labels_losses = self.multi_label_dice_loss(prediction_g, labels_g, self.num_classes, weights)
        l1Loss_g = self.multi_label_l1_loss(prediction_g, labels_g, self.num_classes) if w_l1 > 0. else 0
        # overlap_penalty = torch.relu(prediction_g.sum(dim=1) - 0.5).sum()/labels_g.sum() if w_penalty > 0. else 0
        # overlap_penalty = torch.relu(prediction_g.sum(dim=1) - w_penalty).sum()/labels_g.sum()
        overlap_penalty = 0
        # penalty = overlap_penalty.mean()

        # 두 손실의 가중치 합을 계산 (가중치는 원하는 값으로 설정할 수 있습니다)

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + input_t.size(0)

        # combinedLoss_g = self.cli_args.bce_alpha * bceLoss_g + (1 - self.cli_args.bce_alpha) * diceLoss_g.mean()
        combinedLoss_g = w_bce * bceLoss_g + w_dice * diceLoss_g.mean() + w_l1 * l1Loss_g + overlap_penalty

        if self.cli_args.strategy == 'fedprox':
            global_model_parameters = [param.data for param in self.segmentation_model_global.parameters()]
            local_model_parameters = [param.data for param in self.segmentation_model.parameters()]
            # Proximal term 계산
            proximal_term = self.calculate_proximal_term(local_model_parameters, global_model_parameters)
            mu = self.mu  # 규제 계수, 상황에 맞게 조정하세요.
            # 손실 함수에 Proximal term 추가
            log.info(f"Fedprox loss: {combinedLoss_g + mu * proximal_term} = {combinedLoss_g}+{mu}*{proximal_term} (fedprox:combinedLoss_g + mu * proximal_term).")
            combinedLoss_g = combinedLoss_g + mu * proximal_term
        else:
            log.info(f"Fedavg loss: combinedLoss_g = {combinedLoss_g} (fedavg:combinedLoss_g).")
            

        with torch.no_grad():
            metrics_g[0, start_ndx:end_ndx] = diceLoss_g
            # metrics_g[1, start_ndx:end_ndx] = combinedLoss_g.repeat(diceLoss_g.shape[0])
            # metrics_g[2, start_ndx:end_ndx] = overlap_penalty
            for idx in range(0, self.num_classes):
                metrics_g[OFFSET_METRIC + idx, start_ndx:end_ndx] = labels_losses[idx, :]

        return combinedLoss_g, latent_variables, label_t

    def main(self):
        log.info(f"Starting {type(self).__name__}: {self.analysis_dict['description']}, {self.cli_args}")

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        best_loss = 1.0

        epoch_ndx = 0 
        epoch_with_cycle = self.cli_args.epochs * (self.cli_args.counter - 1) + epoch_ndx - self.cli_args.offset
        # valMetrics_t = self.doValidation(0, train_dl)
        # self.logMetrics(epoch_with_cycle, 'trn', valMetrics_t, self.trn_writer)
        # self.saveModel(self.cli_args.group, self.cli_args.project, self.cli_args.server, 0, False)

        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            epoch_with_cycle = self.cli_args.epochs * (self.cli_args.counter - 1) + epoch_ndx - self.cli_args.offset
            log.info(f"Epoch {epoch_ndx}, LR={self.scheduler.get_last_lr()[0]} of {self.cli_args.epochs}, {len(train_dl)}/{len(val_dl)} batches of size {self.cli_args.batch_size}*{(self.selected_gpu_count if self.use_cuda else 1)}")
            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_with_cycle, 'trn', trnMetrics_t, self.trn_writer)

            if (epoch_ndx == 1) or (epoch_ndx == self.cli_args.epochs) or (epoch_ndx % self.cli_args.validation_cadence == 0):
                valMetrics_t = self.doValidation(epoch_ndx, val_dl)
                mean_loss = self.logMetrics(epoch_with_cycle, 'val', valMetrics_t, self.val_writer)
                self.saveModel(self.cli_args.group, self.cli_args.project, self.cli_args.server, epoch_ndx, valMetrics_t, mean_loss == best_loss)
                best_loss = min(mean_loss, best_loss)

                if epoch_with_cycle == 1:
                    self.logImages(epoch_with_cycle, '1_1_ref_trn', train_dl, 0, self.ref_t_writer)
                    self.logImages(epoch_with_cycle, '2_1_ref_val', val_dl, 1, self.ref_v_writer)
                self.logImages(epoch_with_cycle, '1_2_prd_trn', train_dl, 0, self.trn_writer)
                self.logImages(epoch_with_cycle, '1_2_th_trn', train_dl, 0, self.trn_writer)
                self.logImages(epoch_with_cycle, '2_2_prd_val', val_dl, 1, self.val_writer)
                self.logImages(epoch_with_cycle, '2_2_th_val', val_dl, 1, self.val_writer)
                # self.logImages(epoch_ndx, '3_1_tsne', train_dl, 0, self.tsne_writer)
            self.scheduler.step()
        self.ref_t_writer.close()
        self.ref_v_writer.close()
        self.trn_writer.close()
        self.val_writer.close()
        # self.tsne_writer.close()

    def logImages(self, epoch_ndx, mode_str, dl, pid, writer):
        self.segmentation_model.eval()
        dat = dl.dataset[pid]
        vol = dat[0].unsqueeze(0).to(self.device, non_blocking=True)
        msk = dat[1].unsqueeze(0).to(self.device, non_blocking=True)
        # vol = dl.dataset[pid][0].unsqueeze(0).to(self.device, non_blocking=True)
        # msk = dl.dataset[pid][1].unsqueeze(0).to(self.device, non_blocking=True)
        p_num = dl.dataset.get_pid(pid)
        c = np.array(vol.shape[-3:]) // 2
        ax_ndx = c[0] - c[0] // 6
        co_ndx = c[1] + c[1] // 6
        sa_ndx_right = c[2] - c[2] // 5
        sa_ndx_left = c[2] + c[2] // 7

        pred_msks = self.segmentation_model(vol)[0][0, :].unsqueeze(1)
        dims = list(range(vol.dim()))
        dims[-3], dims[-2], dims[-1] = dims[-2], dims[-3], dims[-1]
        # 먼저, 모든 pred_msk에 대한 처리를 GPU에서 완료합니다.
        all_msk = torch.sigmoid(pred_msks.float())
        all_msk = all_msk.permute(*dims)
        msk = msk.permute(*dims)
        ax_msk = msk[0, :, ax_ndx, :, :].detach().to('cpu')
        co_msk = msk[0, :, :, co_ndx, :].detach().to('cpu')
        sa_msk_R = msk[0, :, :, :, sa_ndx_right].detach().to('cpu')
        sa_msk_L = msk[0, :, :, :, sa_ndx_left].detach().to('cpu')

        ax_msk_color = torch.stack(
            [torch.tensor(self.NORM_COLORDICT.get(i.item(), [0, 0, 0])) for i in ax_msk.flatten()], dim=0).reshape(
            ax_msk.shape[-2], ax_msk.shape[-1], 3).permute(2, 0, 1)
        co_msk_color = torch.stack(
            [torch.tensor(self.NORM_COLORDICT.get(i.item(), [0, 0, 0])) for i in co_msk.flatten()], dim=0).reshape(
            co_msk.shape[-2], co_msk.shape[-1], 3).permute(2, 0, 1)
        sa_msk_color_R = torch.stack(
            [torch.tensor(self.NORM_COLORDICT.get(i.item(), [0, 0, 0])) for i in sa_msk_R.flatten()],
            dim=0).reshape(sa_msk_R.shape[-2], sa_msk_R.shape[-1], 3).permute(2, 0, 1)
        sa_msk_color_L = torch.stack(
            [torch.tensor(self.NORM_COLORDICT.get(i.item(), [0, 0, 0])) for i in sa_msk_L.flatten()],
            dim=0).reshape(sa_msk_L.shape[-2], sa_msk_L.shape[-1], 3).permute(2, 0, 1)

        if mode_str in ['1_1_ref_trn', '2_1_ref_val']:
            vol = vol.permute(*dims)
            # vol
            min_vol, max_vol = torch.min(vol).to('cpu'), torch.max(vol).to('cpu')
            ax_vol = vol[0, :, ax_ndx, :, :].detach().to('cpu')
            co_vol = vol[0, :, :, co_ndx, :].detach().to('cpu')
            sa_vol_R = vol[0, :, :, :, sa_ndx_right].detach().to('cpu')
            sa_vol_L = vol[0, :, :, :, sa_ndx_left].detach().to('cpu')
            reg_ax_vol = (ax_vol - min_vol) / (max_vol - min_vol)
            reg_co_vol = (co_vol - min_vol) / (max_vol - min_vol)
            reg_sa_vol_R = (sa_vol_R - min_vol) / (max_vol - min_vol)
            reg_sa_vol_L = (sa_vol_L - min_vol) / (max_vol - min_vol)
            writer.add_image(f'{mode_str}/{p_num}_img_ax_{ax_ndx}', reg_ax_vol, epoch_ndx)
            writer.add_image(f'{mode_str}/{p_num}_img_co_{co_ndx}', reg_co_vol, epoch_ndx)
            writer.add_image(f'{mode_str}/{p_num}_img_sa_R_{sa_ndx_right}', reg_sa_vol_R, epoch_ndx)
            writer.add_image(f'{mode_str}/{p_num}_img_sa_L_{sa_ndx_left}', reg_sa_vol_L, epoch_ndx)

            writer.add_image(f'{mode_str}/{p_num}_msk_ax_{ax_ndx}', ax_msk_color, epoch_ndx)
            writer.add_image(f'{mode_str}/{p_num}_msk_co_{co_ndx}', co_msk_color, epoch_ndx)
            writer.add_image(f'{mode_str}/{p_num}_msk_sa_R_{sa_ndx_right}', sa_msk_color_R, epoch_ndx)
            writer.add_image(f'{mode_str}/{p_num}_msk_sa_L_{sa_ndx_left}', sa_msk_color_L, epoch_ndx)
        else:

            all_ax_msk = all_msk[:, :, ax_ndx, :, :]
            all_co_msk = all_msk[:, :, :, co_ndx, :]
            all_sa_msk_R = all_msk[:, :, :, :, sa_ndx_right]
            all_sa_msk_L = all_msk[:, :, :, :, sa_ndx_left]
            #
            # all_ax_min_msk, all_ax_max_msk = torch.min(all_ax_msk), torch.max(all_ax_msk)
            # all_co_min_msk, all_co_max_msk = torch.min(all_co_msk), torch.max(all_co_msk)
            # all_sa_R_min_msk, all_sa_R_max_msk = torch.min(all_sa_msk_R), torch.max(all_sa_msk_R)
            # all_sa_L_min_msk, all_sa_L_max_msk = torch.min(all_sa_msk_L), torch.max(all_sa_msk_L)
            # # # 이제 CPU로 옮기고 각 mask를 처리합니다.
            # all_ax_msk = ((all_ax_msk - all_ax_min_msk) / (all_ax_max_msk - all_ax_min_msk)).detach().to('cpu')
            # all_co_msk = ((all_co_msk - all_co_min_msk) / (all_co_max_msk - all_co_min_msk)).detach().to('cpu')
            # all_sa_msk_R = ((all_sa_msk_R - all_sa_R_min_msk) / (all_sa_R_max_msk - all_sa_R_min_msk)).detach().to('cpu')
            # all_sa_msk_L = ((all_sa_msk_L - all_sa_L_min_msk) / (all_sa_L_max_msk - all_sa_L_min_msk)).detach().to('cpu')

            for idx in range(len(all_msk)):
                # label = dl.dataset.class_values[idx]
                label_idx = self.index_classes[idx]
                label_name = CLASSES[label_idx]

                # all_ax_min_msk, all_ax_max_msk = torch.min(all_ax_msk[idx]), torch.max(all_ax_msk[idx])
                # all_co_min_msk, all_co_max_msk = torch.min(all_co_msk[idx]), torch.max(all_co_msk[idx])
                # all_sa_R_min_msk, all_sa_R_max_msk = torch.min(all_sa_msk_R[idx]), torch.max(all_sa_msk_R[idx])
                # all_sa_L_min_msk, all_sa_L_max_msk = torch.min(all_sa_msk_L[idx]), torch.max(all_sa_msk_L[idx])
                # # 이제 CPU로 옮기고 각 mask를 처리합니다.
                # all_ax_msk_reg = ((all_ax_msk[idx] - all_ax_min_msk) / (all_ax_max_msk - all_ax_min_msk)).detach().to('cpu')
                # all_co_msk_reg = ((all_co_msk[idx] - all_co_min_msk) / (all_co_max_msk - all_co_min_msk)).detach().to('cpu')
                # all_sa_msk_R_reg = ((all_sa_msk_R[idx] - all_sa_R_min_msk) / (all_sa_R_max_msk - all_sa_R_min_msk)).detach().to('cpu')
                # all_sa_msk_L_reg = ((all_sa_msk_L[idx] - all_sa_L_min_msk) / (all_sa_L_max_msk - all_sa_L_min_msk)).detach().to('cpu')
                # # all_ax_msk_reg = all_ax_msk[idx]
                # # all_co_msk_reg = all_co_msk[idx]
                # # all_sa_msk_R_reg = all_sa_msk_R[idx]
                # # all_sa_msk_L_reg = all_sa_msk_L[idx]

                if mode_str in ['1_2_prd_trn', '2_2_prd_val']:
                    writer.add_image(f'{mode_str}/{p_num}_prd_ax_{label_idx}_{label_name}_{ax_ndx}', all_ax_msk[idx], epoch_ndx)
                    writer.add_image(f'{mode_str}/{p_num}_prd_co_{label_idx}_{label_name}_{co_ndx}', all_co_msk[idx], epoch_ndx)
                    writer.add_image(f'{mode_str}/{p_num}_prd_sa_R_{label_idx}_{label_name}_{sa_ndx_right}',all_sa_msk_R[idx], epoch_ndx)
                    writer.add_image(f'{mode_str}/{p_num}_prd_sa_L_{label_idx}_{label_name}_{sa_ndx_left}',all_sa_msk_L[idx], epoch_ndx)
                # if mode_str in ['1_2_th_trn', '2_2_th_val']:
                #     writer.add_image(f'{mode_str}/{p_num}_thr_ax_{label_idx}_{label_name}_{ax_ndx}',all_ax_msk_reg >= 0.5, epoch_ndx)
                #     writer.add_image(f'{mode_str}/{p_num}_thr_co_{label_idx}_{label_name}_{co_ndx}',all_co_msk_reg >= 0.5, epoch_ndx)
                #     writer.add_image(f'{mode_str}/{p_num}_thr_sa_R_{label_idx}_{label_name}_{sa_ndx_right}',all_sa_msk_R_reg >= 0.5, epoch_ndx)
                #     writer.add_image(f'{mode_str}/{p_num}_thr_sa_L_{label_idx}_{label_name}_{sa_ndx_left}',all_sa_msk_L_reg >= 0.5, epoch_ndx)
            if mode_str in ['1_2_th_trn', '2_2_th_val']:
                # tmp = all_msk.max(dim=0)[0].cpu().detach().numpy()
                tmp = (torch.clamp(
                    all_msk.max(dim=0)[0], 
                    0, 
                    0.2
                )/0.2).cpu().detach().numpy()

                tmp_ax = tmp[:,ax_ndx,:,:].squeeze(0)
                tmp_co = tmp[:,:,co_ndx,:].squeeze(0)
                tmp_sa_R = tmp[:,:,:,sa_ndx_right].squeeze(0)
                tmp_sa_L = tmp[:,:,:,sa_ndx_left].squeeze(0)

                # ax_msk_color
                # co_msk_color
                # sa_msk_color_R
                # sa_msk_color_L
                # 마스크 합성
                combined_mask_ax = self.combine_masks(ax_msk_color, tmp_ax)
                combined_mask_co = self.combine_masks(co_msk_color, tmp_co)
                combined_mask_sa_r = self.combine_masks(sa_msk_color_R, tmp_sa_R)
                combined_mask_sa_l = self.combine_masks(sa_msk_color_L, tmp_sa_L)

                writer.add_image(f'{mode_str}/{p_num}_ax_all',combined_mask_ax, epoch_ndx)
                writer.add_image(f'{mode_str}/{p_num}_co_all',combined_mask_co, epoch_ndx)
                writer.add_image(f'{mode_str}/{p_num}_sa_R_all',combined_mask_sa_r, epoch_ndx)
                writer.add_image(f'{mode_str}/{p_num}_sa_L_all',combined_mask_sa_l, epoch_ndx)
        writer.flush()

    def logMetrics(self, epoch_ndx, mode_str, metrics_t, writer):
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        metrics_a = metrics_t.detach().numpy()
        # sum_a = metrics_a.sum(axis=1)
        assert np.isfinite(metrics_a).all()

        # allLabel_count = sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_a[0].mean()
        # metrics_dict['loss/all'] = metrics_a[1].mean()
        # metrics_dict['loss/all_penalty'] = metrics_a[2].mean()
        for idx in range(self.num_classes):
            metrics_dict[f'loss/{self.index_classes[idx]}_{CLASSES[self.index_classes[idx]]}'] = metrics_a[OFFSET_METRIC + idx].mean()

        log.info(("E{} {:8} "
                  + "{loss/all:.4f} loss, "
                  + f"{self.cli_args.counter} cycle, "
                  # + "{pr/recall:.4f} recall, "
                  # + "{pr/f1_score:.4f} f1 score"
                  ).format(
            epoch_ndx,
            mode_str,
            **metrics_dict,
        ))

        prefix_str = '0_1_'

        for key, value in metrics_dict.items():
            writer.add_scalar(prefix_str + key, value, epoch_ndx)

        writer.flush()

        # score = metrics_dict['pr/recall']

        return metrics_dict['loss/all']

    def pad(self, img, target_size):
        pad = np.array(target_size) - np.array(img.shape)
        img = np.pad(img, [(pad[0] // 2, pad[0] - pad[0] // 2), (pad[1] // 2, pad[1] - pad[1] // 2)])
        return img

    def combine_masks(self, target, prediction, alpha=0.5):
        # 흑백(prediction) 이미지를 컬러 형식으로 변환
        prediction_color = np.stack((prediction,) * 3, axis=0)

        # 두 이미지 합성
        combined = (target * alpha + prediction_color * (1 - alpha))
        return combined

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.group, self.cli_args.project, self.cli_args.server)

            self.ref_t_writer = SummaryWriter(
                log_dir=log_dir + '_ref_trn')
            self.ref_v_writer = SummaryWriter(
                log_dir=log_dir + '_ref_val')
            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '_prd_trn')
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '_prd_val')
            # SummaryWriter를 생성합니다.
            # self.tsne_writer = SummaryWriter(
            #     log_dir=log_dir + '_tsne_' + self.cli_args.comment)


    def saveModel(self, group, project, server, epoch_ndx, metrics_t, isBest=False):
        
        metrics_a = metrics_t.detach().numpy()
        assert np.isfinite(metrics_a).all()
        loss_list = []
        for idx in range(self.num_classes):
            loss_list.append(metrics_a[OFFSET_METRIC + idx].mean())

        model = self.segmentation_model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        file_path = os.path.join(
            'models',
            group,
            project,
            # self.time_str,
            # f'{type(model).__name__}_{self.time_str}.{epoch_ndx}.state'
            # f'{group}_{project}_{server}.{self.cli_args.counter}.state'
            f'{group}_{server}_{self.cli_args.counter}.state'
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        state = {
            # 'sys_argv': sys.argv,
            # 'time': str(datetime.datetime.now()),
            'losses': loss_list,
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            # 'optimizer_state': self.optimizer.state_dict(),
            # 'optimizer_name': type(self.optimizer).__name__,
            # 'epoch': epoch_ndx,
            # 'totalTrainingSamples_count': self.totalTrainingSamples_count,
        }

        torch.save(state, file_path)

        log.info(f"Saved model params to {file_path}")

        if isBest:
            best_path = os.path.join(
                'models',
                group,
                project,
                # self.time_str,
                # f'{type(model).__name__}_{self.time_str}.{epoch_ndx}.state'
                f'{group}_{server}_{type(model).__name__}.best.state'
                # server,
                # f'{type(model).__name__}_{self.time_str}.best.state'
            )
            shutil.copyfile(file_path, best_path)

            log.info("Saved model params to {}".format(best_path))

        # with open(file_path, 'rb') as f:
        #     log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())



OFFSET_METRIC = 1

if __name__ == '__main__':
    # 현재 프로세스를 새로운 세션의 리더로 설정
    # os.setsid()
    # atexit를 사용해 프로그램 종료시 로그를 남기는 함수
    def exit_handler():
        # 현재 프로세스가 메인 프로세스인 경우
        if os.getpid() == main_pid:
            log.info(f"Main-Program: [PID:{main_pid}] is exiting...")
            # os.killpg(os.getpgid(main_pid), signal.SIGTERM)  # 세션의 모든 프로세스를 종료
        else:
            log.info(f"Child-Workers: [PID:{os.getpid()}] is exiting...")


    ctime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    logconf.setup_logging(ctime)
    atexit.register(exit_handler)
    
    try:
        parser = argparse.ArgumentParser(description="Load model in GainApp class")
        parser.add_argument('-G',   '--gpus',         default=None,    type=str, help='Comma-separated list of GPU device IDs to use (e.g., "0,1" for using GPU 0 and 1), leave empty to use all available GPUs')
        parser.add_argument('-OFF', '--offset',      default=0,       type=int,            help='starting offset for epoch_cycle.')
        parser.add_argument('-CNT', '--counter',      default=0,       type=int,            help='Select initial segmentation model.')
        parser.add_argument('-ANL', '--analysis',      default=1,       type=int,            help='(1) basic', )
        parser.add_argument('-NWK', '--num_workers',   default=4,       type=int,            help='Number of worker processes for background data loading', )
        parser.add_argument('-VLC', '--validation_cadence',default=5,   type=int,            help='Number of epochs to save model and validation for', )
        parser.add_argument('-EPC', '--epochs',        default=1,       type=int,            help='Number of epochs to train for', )
        parser.add_argument('-BAT', '--batch_size', default=1, type=int, help='Batch size to use for training', )

        parser.add_argument('-S', '--seed',            default=1,       type=int,            help='random seed (default: 1)', )
        parser.add_argument('-LR', '--lr',             default=1e-3,    type=float,          help='lr value for Adam optimizer', )
        parser.add_argument('-SS', '--scheduler_step', default=1,       type=int,            help='scheduler step for optimizer', )
        parser.add_argument('-SG', '--scheduler_gamma',default=1.0,       type=float,          help='scheduler gamma for optimizer', )
        parser.add_argument('-B', '--w_bce',           default=0,       type=float,          help='weight for bce-loss', )
        parser.add_argument('-D', '--w_dice',          default=1,       type=float,          help='weight for dice-loss', )
        parser.add_argument('-L', '--w_l1',            default=0,       type=float,          help='weight for l1-loss', )
        parser.add_argument('-P', '--w_penalty',       default=0,       type=float,          help='weight for overlay penalty', )
        parser.add_argument('-E', '--epsilon',         default=1e-5,    type=float,          help='Epsilon value for Dice Loss', )

        parser.add_argument('-PR','--preprocess',      default=False,   action='store_true', help="Preprocessing all data by skullstriping", )
        parser.add_argument('-RS','--resolution',      default=128,     type=int,            help='Pixel size to use for training', )
        parser.add_argument('-CN','--canonical',       default=True,    action='store_true', help="Augment the training data by canonical", )
        parser.add_argument('-DF','--deformation',     default=0.0,     type=float,          help="Augment the training data by deformation", )
        parser.add_argument('-BF','--biasfield',       default=0.0,     type=float,          help="Augment the training data by biasfield", )
        parser.add_argument('-NS','--noise',           default=0.0,     type=float,          help="Augment the training data by noise", )
        parser.add_argument('-FL','--flip',            default=0.0,     type=float,          help="Augment the training data by flip", )
        parser.add_argument('-AF','--affine',          default=0.0,     type=float,          help="Augment the training data by affine", )
        parser.add_argument('-ZM','--zoom',            default=1.0,     type=float,          help="Augment the training data by zoom", )
        parser.add_argument('-ZN','--znorm',           default=True,    action='store_true', help="Augment the training data by znorm", )

        parser.add_argument('-GRP', '--group',         default='none_group',                 help="Group folder for seperator.", )
        parser.add_argument('-PRJ', '--project',       default='none_project',               help="Project folder for seperator.", )
        parser.add_argument('-STG', '--strategy',      default='fedavg',                     help="Select federation strategy. [fedavg, fedprox]", )
        parser.add_argument('-SVR', '--server',        default='none_server',                help="Server name for seperator.", )

        parser.add_argument('comment',                 default='none',  nargs='?',           help="Comment suffix for Tensorboard run.", )
        cli_args = parser.parse_args(sys.argv[1:])
        GaainApp(cli_args).main()
    except Exception as e:
        log.error(f"Error occurred: {e}")
        raise

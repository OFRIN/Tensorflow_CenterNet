# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

# dataset parameters
ROOT_DIR = 'D:/_DeepLearning_DB/COCO/'
TRAIN_DIR = ROOT_DIR + 'train2017/image/'
VALID_DIR = ROOT_DIR + 'valid2017/image/'

CLASS_NAMES = [class_name.strip() for class_name in open('./coco/label_names.txt').readlines()]
CLASS_DIC = {class_name : index for index, class_name in enumerate(CLASS_NAMES)}
CLASSES = len(CLASS_NAMES)

# network parameters
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
IMAGE_CHANNEL = 3

STRIDE = 4 # R

OUTPUT_WIDTH = IMAGE_WIDTH // STRIDE
OUTPUT_HEIGHT = IMAGE_HEIGHT // STRIDE

# ResNet (Normalize), OpenCV BGR -> RGB
R_MEAN = 123.68
G_MEAN = 116.78
B_MEAN = 103.94
MEAN = [R_MEAN, G_MEAN, B_MEAN]

# loss parameters
WEIGHT_DECAY = 0.0001

# train

# multi gpu training
GPU_INFO = "0,1,2,3"
NUM_GPU = len(GPU_INFO.split(','))

BATCH_SIZE = 16 * NUM_GPU
INIT_LEARNING_RATE = 5e-4

# use thread (Dataset)
NUM_THREADS = 10

# iteration & learning rate schedule
MAX_EPOCH = 140 + 90 + 120
DECAY_ITERATIONS = [140, 140 + 90]

LOG_ITERATION = 50
SAMPLE_ITERATION = 10000
SAVE_ITERATION = 10000

# color_list (OpenCV - BGR)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)

COLOR_PBLUE = (204, 72, 63)
COLOR_ORANGE = (0, 128, 255)

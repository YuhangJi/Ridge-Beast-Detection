from utils.kmeans import COCO
# 输入图像
IMAGE_HEIGHT = 640
IMAGE_WIDTH = 640
CHANNELS = 3

# DIR
TRAIN_DIR = r"D:\jiyuhang\new_experiments\ss_detection\data_process\train.txt"
TEST_DIR = r"D:\jiyuhang\new_experiments\ss_detection\data_process\test.txt"
FONT_DIR = r"D:\jiyuhang\new_experiments\ss_detection\font\simsun.ttc"  # constant

# Anchor
CATEGORY_NUM = 14
ANCHOR_NUM_EACH_SCALE = 3
ANCHOR_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
ANCHOR_NUM = ANCHOR_NUM_EACH_SCALE*len(ANCHOR_MASK)
SCALE_SIZE = [13, 26, 52]
AUTO_ANCHORS = False  # if you want to get anchors through k-means algorithm, it should be True
COCO_ANCHORS = COCO(TRAIN_DIR, ANCHOR_NUM, IMAGE_WIDTH, IMAGE_HEIGHT, True).anchors if AUTO_ANCHORS else \
    [[14, 17], [21, 24], [28, 32], [35, 42], [46, 52], [64, 64], [86, 88], [139, 152], [327, 354]]

# Loss
IGNORE_THRESHOLD = 0.50  # constant

# Dataset
BATCH_SIZE = 4
RANDOM_DATA = True

# Training
EPOCHS = 100
LOG_DIR = "./logs"
SAVE_MODEL_DIR = "./logs/net"
MAX_TRUE_BOX_NUM_PER_IMG = 42
SAVE_FREQUENCY = 20
TEST_FREQUENCY = 1

# Resuming Training
IS_LOAD_WEIGHTS = False
START_STEP = None

# Lr_schedule
INITIAL_LEARNING_RATE = 0.0001
DECAY_RATE = 0.50
DECAY_STEP = int(546*18*BATCH_SIZE/4)

# NMS
CONFIDENCE_THRESHOLD = 0.10
IOU_THRESHOLD = 0.50
MAX_BOX_NUM = 36  # recommend value in this dataset

CATEGORY_DICTIONARY = {
        0: "仙人", 1: "龙", 2: "凤", 3: "狮子", 4: "海马", 5: "天马", 6: "狎鱼",
        7: "狻猊", 8: "獬豸", 9: "斗牛", 10: "行什", 11: "垂兽", 12: "吻兽",
        13: "套兽"
    }

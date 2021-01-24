import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from yolo.yolo_loss import preprocess_true_boxes
from config import BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CATEGORY_NUM, MAX_BOX_NUM, RANDOM_DATA


def getRandNumber(a=None, b=None):
    if (a is None) and (b is None):
        a = 0
        b = 1
    return np.random.rand() * (b - a) + a


def getLengthOfDataset(dataset):
    count = 0
    for _ in dataset:
        count += 1
    return count


def sparseLineData(line_data,training, is_random=RANDOM_DATA, is_proc_img=True, jitter=.3, hue=.1, sat=1.5, val=1.5):
    line_data = bytes.decode(line_data.numpy(), encoding="utf-8")
    line_data = line_data.strip()
    line_data = line_data.split(' ')
    img_data = Image.open(line_data[0])
    img_w, img_h = img_data.size
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line_data[1:]])
    if training and is_random:
        # resize image
        new_ar = IMAGE_WIDTH / IMAGE_HEIGHT * getRandNumber(1 - jitter, 1 + jitter) / getRandNumber(1 - jitter,
                                                                                                    1 + jitter)
        scale = getRandNumber(.25, 2)
        if new_ar < 1:
            nh = int(scale * IMAGE_HEIGHT)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * IMAGE_WIDTH)
            nh = int(nw / new_ar)
        image = img_data.resize((nw, nh), Image.BICUBIC)

        # place image
        dx = int(getRandNumber(0, IMAGE_WIDTH - nw))
        dy = int(getRandNumber(0, IMAGE_HEIGHT - nh))
        new_image = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = getRandNumber() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = getRandNumber(-hue, hue)  # hue .1
        sat = getRandNumber(1, sat) if getRandNumber() < .5 else 1 / getRandNumber(1, sat)  # 1.5
        val = getRandNumber(1, val) if getRandNumber() < .5 else 1 / getRandNumber(1, val)  # 1.5
        x = rgb_to_hsv(np.array(image) / 255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        img_data = hsv_to_rgb(x)  # numpy array, 0 to 1

        # correct boxes
        box_data = np.zeros((MAX_BOX_NUM, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / img_w + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / img_h + dy
            if flip: box[:, [0, 2]] = IMAGE_WIDTH - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > IMAGE_WIDTH] = IMAGE_WIDTH
            box[:, 3][box[:, 3] > IMAGE_HEIGHT] = IMAGE_HEIGHT
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            if len(box) > MAX_BOX_NUM: box = box[:MAX_BOX_NUM]
            box_data[:len(box)] = box
    else:
        # resize image
        scale = min(IMAGE_WIDTH / img_w, IMAGE_HEIGHT / img_h)
        nw = int(img_w * scale)
        nh = int(img_h * scale)
        dx = (IMAGE_WIDTH - nw) // 2
        dy = (IMAGE_HEIGHT - nh) // 2
        if is_proc_img:
            img_data = img_data.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), (128, 128, 128))
            new_image.paste(img_data, (dx, dy))
            img_data = np.array(new_image, dtype=np.float32) / 255.

        # correct boxes
        box_data = np.zeros((MAX_BOX_NUM, 5))
        if len(box) > 0:
            if training:
                np.random.shuffle(box)
            if len(box) > MAX_BOX_NUM: box = box[:MAX_BOX_NUM]
            box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
            box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
            box_data[:len(box)] = box
    img_data = np.expand_dims(img_data, axis=0)
    return img_data, box_data


def sparseBatchData(batch_data,training):
    batch_image = []
    batch_boxes = []
    for one_line_data in batch_data:
        one_image, one_boxes = sparseLineData(one_line_data,training)
        batch_image.append(one_image)
        batch_boxes.append(one_boxes)
    batch_image = np.array(batch_image, dtype=np.float32)
    batch_image = np.concatenate(batch_image, axis=0)
    batch_boxes = np.array(batch_boxes)
    # Generate labels
    batch_labels = preprocess_true_boxes(true_boxes=batch_boxes,
                                         input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                         num_classes=CATEGORY_NUM)

    return batch_image, batch_labels[0], batch_labels[1], batch_labels[2]


def generateDataset(txt_dir,training):
    ds = tf.data.TextLineDataset(txt_dir)
    ds_length = getLengthOfDataset(ds)
    ds = ds.shuffle(ds_length)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.map(
        lambda x: tf.py_function(sparseBatchData,
                                 inp=[x,training],
                                 Tout=[tf.float32, tf.float32, tf.float32, tf.float32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    ds = ds.prefetch(BATCH_SIZE)
    print("Load {} data!".format(ds_length))
    return ds, ds_length


def generateDatasetIterator(txt_dir,training):
    ds, ds_length = generateDataset(txt_dir,training)
    ds = ds.repeat()
    ds_iterator = ds.as_numpy_iterator()
    return ds_iterator, ds_length

# using demo
# dataset = generate_dataset()
# for images,labels1,labels2,labels3 in dataset:
#     data = images
#     label = [labels1,labels2,labels3]

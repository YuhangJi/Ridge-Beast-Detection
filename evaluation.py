import os
import time
import shutil
import numpy as np
from PIL import Image
from yolo.yolo_loss import yolo_eval
from utils.tools import get_separator
from config import (
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    CHANNELS,
    CATEGORY_DICTIONARY,
    CATEGORY_NUM
)

total_time = 0.


def letterbox_image(image):
    """resize image with unchanged aspect ratio using padding"""
    iw, ih = image.size
    w = IMAGE_WIDTH
    h = IMAGE_HEIGHT
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def inferenceSingleImage(orig_img, model):
    """
    The annotation record the coordinates of boxes on original images, and the predicted boxes restored original too,
    but its coordinates is reversed, for example, you need to call box[1],box[0],box[3],box[2] to describe a objective
    coordinates like boxes you have labeled instead of box[0],box[1],box[2],box[3].
    """
    global total_time

    resized_img = letterbox_image(orig_img)
    img_w, img_h = orig_img.size
    resized_img = np.expand_dims(np.array(resized_img, dtype=np.float32) / 255., axis=0)

    # inference time >>>
    time_a = time.time()
    yolo_outputs = model(resized_img, training=False)
    time_b = time.time()
    total_time += time_b - time_a
    # <<<

    pred_boxes, pred_scores, pred_classes = yolo_eval(
        yolo_outputs=yolo_outputs,
        image_h=img_h,
        image_w=img_w
    )
    return pred_boxes.numpy(), pred_scores.numpy(), pred_classes.numpy()


def createDRFiles(test_txt_dir, save_dir, model):
    """
    Create detection results files
    """

    if os.path.exists(save_dir): shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # Get Image Path
    img_path_list = [line.split(' ')[0] for line in open(test_txt_dir, 'r').readlines()]

    for img_path in img_path_list:
        name = img_path.split(get_separator())[-1]
        file_name = name.replace(name[-3:], 'txt')
        img_data = Image.open(img_path)
        # Inference Single Image
        boxes, scores, classes = inferenceSingleImage(orig_img=img_data, model=model)
        with open(os.path.join(save_dir, file_name), 'w') as fo:
            for i in range(classes.size):
                fo.write(
                    ' '.join(
                        [CATEGORY_DICTIONARY[int(classes[i])],
                         str(scores[i]),
                         str(int(boxes[i, 1])),
                         str(int(boxes[i, 0])),
                         str(int(boxes[i, 3])),
                         str(int(boxes[i, 2]))]
                    ) + '\n'
                )
    print("Average time of inferencing single image is {:.4f} seconds.".format(total_time / len(img_path_list)))


def createGTFiles(path, save_dir, mode=1):
    """
    mode 1 create from test.txt
    mode 0 create from xml
    """
    if os.path.exists(save_dir): shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    if mode == 1:
        for i in [f.strip() for f in open(path, 'r').readlines()]:
            line = i.split(' ')
            name = line[0].split(get_separator())[-1]  # picture name
            file_name = name.replace(name[-3:], 'txt')  # txt name
            objectives = line[1:]  # all objectives in one image
            # write ground trues file
            with open(os.path.join(save_dir, file_name), 'w') as fo:
                for j in objectives:
                    left, top, right, bottom, category = j.split(',')
                    fo.write(
                        ' '.join([str(CATEGORY_DICTIONARY[int(category)]), left, top, right, bottom]) + '\n'
                    )
    else:
        print("waiting for implementing")


def get_map(test_file_path, gt_file_path, dr_file_path, model):
    createGTFiles(test_file_path, gt_file_path)
    createDRFiles(test_file_path, dr_file_path, model)
    import mAP.calculate_map


if __name__ == '__main__':

    from yolo.model import Net
    weights_path = r'D:\jiyuhang\new_experiments\ss_detection\logs\yolov3'
    net = Net(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS),
                 out_channels=3 * (CATEGORY_NUM + 5),
                 alpha=0.1)
    net.build()
    net.summary()
    net.load_weights(filepath=weights_path)
    get_map(test_file_path=r'D:\jiyuhang\new_experiments\ss_detection\data_process\test.txt',
            gt_file_path=r'D:\jiyuhang\new_experiments\ss_detection\mAP\input\ground-truth',
            dr_file_path=r'D:\jiyuhang\new_experiments\ss_detection\mAP\input\detection-results',
            model=net)



import os
import time
import shutil
import cv2
import numpy as np
from utils.tools import object_statistics, get_separator, match_postfix
from PIL import Image
from yolo.yolo_loss import yolo_eval
from config import IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS, \
    mAP_IOU_THRESHOLD, CATEGORY_DICTIONARY, CATEGORY_NUM, TEST_DIR, IS_TINY, BACKBONE
from matplotlib import pyplot as plt
from utils.visulization import drawBox

total_time = 0.  # 用于计算平均单张图像推理时间


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


def evaluateSingleImage(orig_img, model, is_drawing=True, is_showing=False, is_saving=True, save_name="set_path.jpg",
                        timer=True, detect_video=False):
    """
    The annotation record the coordinates of boxes on original images, and the predicted boxes restored original too, but
    its coordinates is reversed, for example, you need to call box[1],box[0],box[3],box[2] to describe a objective
    coordinates like boxes you have labeled instead of box[0],box[1],box[2],box[3].
    """
    global total_time
    time_a = time.time()
    resized_img = letterbox_image(orig_img)
    img_w, img_h = orig_img.size
    pred_boxes, pred_scores, pred_classes = yolo_eval(
        yolo_outputs=model(np.expand_dims(np.array(resized_img, dtype=np.float32) / 255., axis=0),training=False),
        image_h=img_h,
        image_w=img_w
    )
    if is_drawing:
        draw_img = drawBox(
            image=orig_img,
            classes=pred_classes.numpy(),
            scores=pred_scores.numpy(),
            boxes=pred_boxes.numpy())
        if is_showing:
            plt.figure("Picture")
            plt.imshow(draw_img)
            plt.show()
        if is_saving:
            draw_img.save(save_name)
        if detect_video:
            return draw_img
    time_b = time.time()
    if timer: total_time += time_b - time_a

    return pred_boxes.numpy(), pred_scores.numpy(), pred_classes.numpy()


def getMAP(object_file_dir, is_visualization=True):
    txt_dict = {_.split(get_separator())[-1][:-4]: _ for _ in match_postfix(object_file_dir, 'txt')}

    # check
    for _ in txt_dict.keys():
        button = True
        for __ in CATEGORY_DICTIONARY.keys():
            if _ == CATEGORY_DICTIONARY[__]:
                button = False
                break
        if button:
            raise ValueError(
                "A conflicting condition has been found between information file and category dictionary. ")

    def getTestImageGT(test_dir):
        """
        Generate a dictionary like {xxx.jpg:gt}. gt is 2D-array [[box1],[box2],[],...] with shape of [N,5].
        Every box has format: xmin, ymin, xmax, ymax, class, such as 100, 200, 300, 400, 1
        :param test_dir:
        :return: dictionary
        """
        test_dict = {}
        for i in [line.strip() for line in open(test_dir, 'r').readlines()]:
            i = i.split(' ')
            img_name = i[0].split(get_separator())[-1]
            boxes = [list(map(lambda x: int(x), j.split(','))) for j in i[1:]]
            test_dict[img_name] = np.array(boxes, dtype=np.int)
        return test_dict

    def calIOU(box_pred, box_gt):
        """
        calculate iou value between single predicted box and all boxes in it's image.
        :param box_pred: array,shape [1,4].
        :param box_gt: array,shape [N,4].
        :return:
        """
        box_pred = np.asarray(box_pred, dtype=np.float32)
        box_gt = np.asarray(box_gt, dtype=np.float32)

        # [1, 2] & [N 2] ==> [N, 2]
        intersect_mins = np.maximum(box_pred[..., :2], box_gt[..., :2])
        intersect_maxs = np.minimum(box_pred[..., 2:], box_gt[..., 2:])
        intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0.)

        # shape: [N]
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        # shape: [1]
        pred_box_wh = box_pred[..., 2:] - box_pred[..., :2]
        # shape: [1]
        pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
        # [N,2]
        true_boxes_wh = box_gt[..., 2:] - box_gt[..., :2]
        # [N]
        true_boxes_area = true_boxes_wh[..., 0] * true_boxes_wh[..., 1]

        # shape: [N]
        iou = intersect_area / (pred_box_area + true_boxes_area - intersect_area + 1e-10)
        return iou

    def getPRCurveXYAxis(info_file_dir,gt_num):
        """
        Input a txt file and output X,Y coordinates. The txt file records all boxes information within one category.
        The file format: xxx.jpg confidence xmin ymin xmax ymax
        And its file name should be categorical name.
        :param info_file_dir:
        :param gt_num:
        :return:
        """
        # get TP/FP list >>>
        pred_lines = [lines.strip() for lines in open(info_file_dir, 'r').readlines()]
        lines_id = []  # each line has a id
        lines_name = []
        lines_confidence = []
        lines_box = []
        n = 0
        for line in pred_lines:
            line = line.split(' ')
            name, confidence, box = line[0], float(line[1]), [int(_) for _ in line[2:6]]
            lines_name.append(name)
            lines_confidence.append(confidence)
            lines_box.append(np.array(box))
            lines_id.append(n)
            n += 1

        tp_fp_list = []

        # judge every box is TP or FP in one txt-file.
        for i in lines_id:
            test_name = lines_name[i]  # the name of test image
            box_line = lines_box[i]  # a line present a box
            gt_boxes = test_gt_dict[test_name][..., :4]  # take all true boxes from image dictionary
            gt_classes = test_gt_dict[test_name][..., -1]
            box_gt_iou = calIOU(box_pred=box_line, box_gt=gt_boxes)
            tem_mask = box_gt_iou >= mAP_IOU_THRESHOLD
            if class_id in gt_classes[tem_mask]:
                tp_fp_list.append(1)
            else:
                tp_fp_list.append(0)
        # <<<

        lines_info = sorted(
            list(zip(lines_id, lines_name, lines_confidence, lines_box, tp_fp_list)),
            key=lambda _: _[2],
            reverse=True
        )

        tp_fp = np.array(list(map(lambda x: int(x), np.asarray(lines_info)[..., -1])), dtype=np.float32)
        recall_value = []
        precision_value = []

        for idx in range(1, tp_fp.size + 1):
            recall_value.append(np.sum(tp_fp[:idx]) / gt_num)
            precision_value.append(np.sum(tp_fp[:idx]) / idx)

        # convert list to array
        precision_value = np.array(precision_value, dtype=np.float32)
        recall_value = np.array(recall_value, dtype=np.float32)

        recall_threshold_x = np.unique(recall_value)
        precision_y = []
        for rec in recall_threshold_x:
            tem_mask = recall_value >= rec
            precision_y.append(np.max(precision_value[tem_mask]))

        precision_y = np.array(precision_y, dtype=np.float32)

        return recall_threshold_x, precision_y, recall_value[-1], precision_value[-1]

    def getAP(x, y):
        """x and y is array."""
        if x.size != y.size:
            raise ValueError("x axis is not equal to y axis.")

        s = 0.
        for index in range(1, x.size):
            ds = (x[index] - x[index - 1]) * y[index]
            if ds < 0:
                raise RuntimeError("Area is negative {} when calculating AP value!".format(ds))
            s += ds
        return s

    test_gt_dict = getTestImageGT(TEST_DIR)
    # 计算测试图像中每个类别一共标注了多少框，用于后面计算召回率。
    every_category_gt_num = np.zeros(shape=(len(txt_dict.keys()),))
    for _ in test_gt_dict.keys():
        for __ in test_gt_dict[_][...,-1]:
            every_category_gt_num[__] += 1
    ap_dict = {}
    plt.figure(num=1)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel("recall")
    plt.ylabel("precision")

    result_dict = {}

    for class_id in CATEGORY_DICTIONARY.keys():
        info_file = txt_dict[CATEGORY_DICTIONARY[class_id]]
        # calculate ap value
        r_x, p_y, r, p = getPRCurveXYAxis(info_file,every_category_gt_num[class_id])
        result_dict[CATEGORY_DICTIONARY[class_id]] = {"r": r, "p": p}
        ap_dict[CATEGORY_DICTIONARY[class_id]] = getAP(r_x, p_y)

        if is_visualization:
            plt.plot(np.insert(np.append(r_x, 1), 0, 0), np.insert(np.append(p_y, 0), 0, 1))

    # average recall and average precision
    total_p, total_r = 0., 0.
    for _ in result_dict.keys():
        print("{} recall: {:.4f} precision: {:.4f} AP:{:.4f} ".format(_, result_dict[_]['r'],
                                                                      result_dict[_]['p'],
                                                                      ap_dict[_])
              )
        total_p += result_dict[_]['p']
        total_r += result_dict[_]['r']
    avg_p = total_p / len(result_dict.keys())
    avg_r = total_r / len(result_dict.keys())
    print("Average recall: {:.4f} precision: {:.4f} ".format(avg_r, avg_p))

    map_ = 0.
    for _ in ap_dict.keys():
        map_ += ap_dict[_]
    map_ /= len(ap_dict.keys())
    print("mAP: {:.4f} ".format(map_))

    plt.show()

    return map_


def inferenceDirImageWriteInfo(test_txt_name, model, is_drawing=True, is_showing=False, is_saving=True, timer=True):
    """
    Inference all images of directory and generate information txt which records each box and confidence of each
    category for one pictures.
        .../information/
            category_1.txt
            category_2.txt
            ...
            category_n.txt

    category_i.txt:
                    name box
                    001.jpg 0.85 20 30 40 50
                    002.jpg 0.65 30 40 50 60
                    ...
    :param timer: Bool
    :param is_showing: Bool
    :param is_drawing: Bool
    :param is_saving: Bool
    :param model:
    :param test_txt_name: The file test.txt
    :return: None, generate a directory, named results, to save result.
    """
    # Delete results run last time
    # Generate Dir
    info_dir = os.path.join("results", "information")
    pic_dir = os.path.join("results", "pictures")
    if os.path.exists("results"): shutil.rmtree("results")
    os.makedirs(info_dir)
    os.makedirs(pic_dir)
    # Statistics
    object_statistics(test_txt_name, CATEGORY_NUM)  # statistic test samples number.
    # Get Image Path
    img_path_list = [line.split(' ')[0] for line in open(test_txt_name, 'r').readlines()]
    # Define Information Dictionary
    info_dict = {CATEGORY_DICTIONARY[_]: {"name": [], "box": []} for _ in range(CATEGORY_NUM)}
    # info_dict: dict_1, dict_2, ..., dict_n  each category has an individual dictionary
    # and dict_i: "name" "box" → dict_i: [] [] each dictionary has two key to restore name and box in individual list.
    for img_path in img_path_list:
        img_name = img_path.split(get_separator())[-1]
        img_data = Image.open(img_path)
        # Inference Single Image
        boxes, scores, classes = evaluateSingleImage(
            orig_img=img_data,
            model=model,
            is_drawing=is_drawing,
            is_showing=is_showing,
            is_saving=is_saving,
            save_name=os.path.join(pic_dir, img_name),
            timer=timer
        )
        # If detect nothing, continue
        if classes.shape == (0,):
            continue
        else:
            # Record each box in this one detected picture.
            for idx in range(len(classes)):
                category_id = int(classes[idx])
                confidence = scores[idx]
                coordinates = boxes[idx]
                info_dict[CATEGORY_DICTIONARY[category_id]]["name"].append(img_name)
                info_dict[CATEGORY_DICTIONARY[category_id]]["box"].append(
                    str(confidence) + ' ' +
                    str(int(coordinates[1])) + ' ' +
                    str(int(coordinates[0])) + ' ' +
                    str(int(coordinates[3])) + ' ' +
                    str(int(coordinates[2])) + ' '
                )
    if timer:
        print("Average time of inferencing single image is {:.4f} seconds.".format(total_time / len(img_path_list)))

    def writeFromInfoDict(info_dictionary, file_save_dir):
        for category_name in list(info_dictionary.keys()):
            names_list = info_dictionary[category_name]["name"]
            boxes_list = info_dictionary[category_name]["box"]
            # Write all boxes belonged to same class into one file.
            with open(os.path.join(file_save_dir, category_name + '.txt'), 'w') as fo:
                for jdx in range(len(names_list)):
                    fo.write(names_list[jdx] + ' ' + boxes_list[jdx] + '\n')

    writeFromInfoDict(info_dict, info_dir)
    """
    demo
    inferenceDirImageWriteInfo(r"D:\jiyuhang\new_experiments\ss_detection\data_process\test.txt", net)
    getMAP(r"D:\jiyuhang\new_experiments\ss_detection\results\information")

    """


def iWannaSeePictures(path,save_path,model):
    img_list = match_postfix(path,'jpg')
    for i in img_list:
        data = Image.open(i)
        name = i.split(get_separator())[-1]
        evaluateSingleImage(
            orig_img=data,
            model=model,
            is_drawing=True,
            is_showing=False,
            is_saving=True,
            save_name=os.path.join(save_path, name),
            timer=True
        )


if __name__ == '__main__':
    # Inference
    # from yolo.models import YOLOV3

    from yolo.selfmodel_selfheadv1 import YOLOV3
    net = YOLOV3(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS),
                 out_channels=3 * (CATEGORY_NUM + 5), backbone='darknet', alpha=0.1)
    net.build()
    net.summary()
    net.load_weights(filepath=r'D:\jiyuhang\new_experiments\ss_detection\logs\selfmodel_selfhead\yolov3epoch-100')

    # video_path = r'C:\Users\jiyuhang\Desktop\rb_video2.mp4'
    # vid = cv2.VideoCapture(video_path)
    # video_fps = int(vid.get(cv2.CAP_PROP_FPS))
    # video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
    #               int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # videoWriter = cv2.VideoWriter('rb_detection_11.avi', cv2.VideoWriter_fourcc(*'MJPG'), video_fps, video_size)
    # print(video_fps,video_size)
    # ret, frame = vid.read()
    # while ret:
    #
    #     image = Image.fromarray(frame[...,::-1])
    #     image = evaluateSingleImage(image,net,True,False,False,'',False,True)
    #     result = np.asarray(image)[...,::-1]
    #     videoWriter.write(result)
    #     ret, frame = vid.read()

    from utils.tools import match_postfix

    videoes = match_postfix(r'C:\Users\jiyuhang\Desktop\test','mp4')
    for i in videoes:
        vid = cv2.VideoCapture(i)
        video_fps = int(vid.get(cv2.CAP_PROP_FPS))
        video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        videoWriter = cv2.VideoWriter(i.replace('mp4','avi'), cv2.VideoWriter_fourcc(*'MJPG'), video_fps, video_size)
        print(video_fps,video_size)
        ret, frame = vid.read()
        while ret:

            image = Image.fromarray(frame[...,::-1])
            image = evaluateSingleImage(image,net,True,False,False,'',False,True)
            result = np.asarray(image)[...,::-1]
            videoWriter.write(result)
            ret, frame = vid.read()





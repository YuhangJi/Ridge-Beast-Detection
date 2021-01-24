from __future__ import division, print_function
import os
import cv2
import numpy as np


def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    param:
        box: tuple or array, shifted to the origin (i. e. width and height)
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = np.true_divide(intersection, box_area + cluster_area - intersection + 1e-10)
    # iou_ = intersection / (box_area + cluster_area - intersection + 1e-10)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    param:
        boxes: numpy array of shape (r, 4)
    return:
    numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        k: number of clusters
        dist: distance function
    return:
        numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters


def parse_anno(annotation_path, target_size=None):
    anno = open(annotation_path, 'r')
    result = []
    for line in anno:
        s = line.strip().split(' ')
        img_w = int(s[2])
        img_h = int(s[3])
        s = s[4:]
        box_cnt = len(s) // 5
        for i in range(box_cnt):
            x_min, y_min, x_max, y_max = float(s[i * 5 + 1]), float(s[i * 5 + 2]), float(s[i * 5 + 3]), float(
                s[i * 5 + 4])
            width = x_max - x_min
            height = y_max - y_min
            assert width > 0
            assert height > 0
            # use letterbox resize, i.e. keep the original aspect ratio
            # get k-means anchors on the resized target image size
            if target_size is not None:
                resize_ratio = min(target_size[0] / img_w, target_size[1] / img_h)
                width *= resize_ratio
                height *= resize_ratio
                result.append([width, height])
            # get k-means anchors on the original image size
            else:
                result.append([width, height])
    result = np.asarray(result)
    return result


def get_kmeans(anno, cluster_num=9):
    anchors = kmeans(anno, cluster_num)
    ave_iou = avg_iou(anno, anchors)

    anchors = anchors.astype('int').tolist()

    anchors = sorted(anchors, key=lambda x: x[0] * x[1])

    return anchors, ave_iou


def gen_kmeans_annotation(train_annotation):
    """
    Convert a standard annotation format to the "get_kmeans" function needed.
    Standard: xxx.jpg 100,200,300,400,1 300,600,500,800,2 ...
                image_name box_1 box_2 ... box_n
                box_i:xmin, ymin, xmax, ymax, class_id
    Needed  : 0 xxx/xxx/a.jpg 1920 1080 0 453 369 473 391 1 588 245 608 268
                image_index image_absolute_path img_width img_height box_1 box_2 ... box_n
                box_i: label_index x_min y_min x_max y_max
    Note: The needed annotation file use whitespace as separator between every value in one box, instead of ','.
    :param train_annotation: The Standard annotation file.
    :return:
    """
    standard_line = [line.strip() for line in open(train_annotation, 'r').readlines()]
    tem_file = "k-means_annotation.txt"
    with open(tem_file,'w') as fo:
        for idx, line in enumerate(standard_line):
            line = line.split(' ')
            image_path = line[0]
            boxes = line[1::]
            image_shape = cv2.imread(image_path).shape  # H W Channel
            # convert class_id last to class_id first for each box
            converted_boxes = []
            for box in boxes:
                box = box.split(',')
                box.insert(0, box[-1])
                box.pop(-1)
                box = ' '.join(box)  # note that this separator is whitespace ' '.
                converted_boxes.append(box)
            converted_boxes = ' '.join(converted_boxes)
            # Write annotation file
            fo.write(str(idx) + ' ' +
                     image_path+' ' +
                     str(image_shape[1]) + ' ' +
                     str(image_shape[0]) + ' ' +
                     converted_boxes + '\n'
                     )
    return tem_file


def get_cluster_anchors(annotation_path, anchor_num, img_width, img_height, needed_resize=True):
    print("------Generating anchors automatically------")
    # target resize format: [width, height]
    # if target_resize is speficied, the anchors are on the resized image scale
    # if target_resize is set to None, the anchors are on the original image scale
    if needed_resize:
        target_size = [img_width, img_height]
    else:
        target_size = None
    # generate the annotation file requested by "parse_anno" function.
    annotation_path = gen_kmeans_annotation(annotation_path)
    # generate anchor
    anno_result = parse_anno(annotation_path, target_size=target_size)
    anchors, ave_iou = get_kmeans(anno_result, anchor_num)
    # anchors = [anchors[i_*3-3:i_*3][j_] for i_ in range(3,0,-1) for j_ in range(0,3,1)]

    anchor_string = ''
    for anchor in anchors:
        anchor_string += '{},{}, '.format(anchor[0], anchor[1])
    anchor_string = anchor_string[:-2]

    print('{} Anchors:'.format(anchor_num))
    print(anchor_string)
    print('The Average Iou:')
    print(ave_iou)
    if os.path.isfile(annotation_path): os.remove(annotation_path)
    print("------Finishing generation------")
    return anchors


class COCO:
    def __init__(self,annotation_file,anchors_count,image_w,image_h,needed_resize=True):
        self.annotation_file = annotation_file
        self.anchors_count = anchors_count
        self.image_w = image_w
        self.image_h = image_h
        self.needed_resize = needed_resize

    def __get_auto_anchors(self):
        return get_cluster_anchors(self.annotation_file,self.anchors_count,self.image_w,self.image_h,self.needed_resize)

    @property
    def anchors(self):
        return self.__get_auto_anchors()


if __name__ == '__main__':
    # The function "get_cluster_anchors" can generate anchors that has been resized according to input size you seted.
    get_cluster_anchors(r"D:\jiyuhang\new_experiments\ss_detection\data_process\train.txt",9,416,416,True)

import os
import platform
import shutil
from PIL import Image
import xml.etree.ElementTree as ET


def get_files(path, followlinks=False):
    """获取某一目录下的文件

    :param followlinks: 是否包含软硬链接
    :param path: 根目录;
    :return:返回文件列表
    """
    root, dirs, files = range(3)
    return [os.path.join(_[root], __) for _ in os.walk(path, followlinks=followlinks) for __ in _[files]]


def ptm():
    """判断系统是windows还是linux

    :return: windows 返回 0 linux 返回 1
    """
    c = -1
    my_os = platform.system()
    if my_os == 'Windows': c = 0
    if my_os == 'Linux': c = 1
    return c


def get_separator():
    if ptm():
        return "/"
    else:
        return "\\"


def match_postfix(path, postfix):
    """匹配路径中指定后缀名的文件

    :param path: 根目录；
    :param postfix: 后缀名,不包含字符“.”；
    :return: 列表。
    """
    file_list = []
    for roots, dirs, files in os.walk(path, followlinks=False):
        for i in files:
            if i.split('.')[-1] == postfix:
                file_list.append(os.path.join(roots, i))
    return file_list


def rm_noobj(path, key='object'):
    """用于yolo检测框架中，对空xml文件的处理。

    :param path: 根目录;
    :param key: 查找字段;
    :return: None
    """
    # >>>创建空xml转移路径
    save_path = '__non-key.xml__'
    if not os.path.exists(save_path): os.mkdir(save_path)
    # <<<

    xml_list = match_postfix(path=path, postfix='xml')
    for i in xml_list:
        tree = ET.parse(i)
        root = tree.getroot()
        obj = root.findall(key)
        if len(obj) == 0: shutil.move(i, save_path)


def rm_error(path, classes_range):
    """用于yolo检测框架中，对错误标注xml的处理。

    :param path: 根目录;
    :param classes_range: 一个元组或者列表,(a,b)---(标注类别最小值,标注类别最大值);
    :return: None
    """

    # >>>创建标注错误的xml转移路径
    save_path = '__error-label.xml__'
    if not os.path.exists(save_path): os.mkdir(save_path)
    # <<<

    # >>>判断参数格式是否为元组或者列表
    if not isinstance(classes_range, (list, tuple)):
        raise ValueError("parameter classes_range is invalid.")
    # <<<

    # 产生期望序列
    classes_range = [str(i) for i in list(range(classes_range[0], classes_range[1], 1))]
    print('Expected Label List : ', classes_range)

    xml_list = match_postfix(path=path, postfix='xml')
    for i in xml_list:
        button = False  # 是否移动该xml的开关按钮
        tree = ET.parse(i)
        root = tree.getroot()
        obj = root.findall('object')
        for j in obj:
            if j.find('name').text not in classes_range:
                button = True
                print('\t', i, 'error label ', '------>', j.find('name').text)
        if button:
            shutil.move(i, save_path)
            print('\t\t\terror label has been moved to {}.'.format(save_path))


def rm_nojpg(path):
    """用于yolo检测框架，处理仅有xml却无图像的xml文件(移到指定目录)。
        注意：需要将xml所匹配的jpg放到同一目录下。

    :param path: 根目录;
    :return: None
    """

    # >>>创建无图像的xml转移路径
    save_path = '__non-img.xml__'
    if not os.path.exists(save_path): os.mkdir(save_path)
    # <<<

    xml_list = match_postfix(path, 'xml')
    jpg_list = match_postfix(path, 'jpg')

    for i in xml_list:
        if i.replace('.xml', '.jpg') not in jpg_list: shutil.move(i, save_path)


def rot(img_path):
    """用于yolo检测框架，给jpg、xml做镜像变换，结果存放在当前目录的rot目录内。

    :param img_path: 根目录
    :return: None
    """

    def cal(x, d):
        return 2 * d - x

    # >>>检测os
    if ptm() == -1:
        print("unknown system , please check code .")
        return None
    # <<<

    file_list = get_files(img_path)
    img_list = []
    xml_list = []
    for i in file_list:
        if i.split('.')[-1] == 'jpg': img_list.append(i)
        if i.split('.')[-1] == 'xml': xml_list.append(i)
    save_path = os.path.join(os.getcwd(), 'rot')
    if not os.path.exists(save_path): os.mkdir(save_path)
    for i in img_list:
        img = Image.open(i)
        img1 = img.transpose(Image.FLIP_LEFT_RIGHT)
        img1.save(os.path.join(save_path, 'mirror_' + i.replace(('\\', '/')[ptm()], '!')))
    for k in xml_list:
        tree = ET.parse(k)
        roots = tree.getroot()
        size = roots.findall('size')
        dt = int(int(size[0].find('width').text) / 2)
        for i in roots.findall('object'):
            for j in i.findall('bndbox'):
                xmin = int(j.find('xmin').text)
                xmax = int(j.find('xmax').text)
                w = xmax - xmin
                j.find('xmin').text = str(cal(xmin, dt) - w)
                j.find('xmax').text = str(cal(xmax, dt) + w)
        filename = 'mirror_' + k.replace(('\\', '/')[ptm()], '!')
        roots.find('filename').text = filename
        tree.write(os.path.join(save_path, filename), encoding="utf-8")


def label_creator(path, classes_range, key='object', save_name="label.txt"):
    """用于yolo检测框架生成训练标签。

    :param save_name: 保存的文件名
    :param path: 根目录;
    :param classes_range: 一个元组或者列表,(a,b)---(标注类别最小值,标注类别最大值);
    :param key: 查找字段;
    :return:None
    """

    # >>>clean data
    rm_noobj(path=path, key=key)  # empty xml
    rm_error(path=path, classes_range=classes_range)  # label error
    rm_nojpg(path=path)  # no jpg
    # <<<

    order = [str(i) for i in list(range(classes_range[0], classes_range[1] + 1, 1))]

    xml_list = match_postfix(path, 'xml')
    label_list = []
    for xml in xml_list:
        jpg = xml.replace('.xml', '.jpg')  # xml路径对应的照片路径
        tree = ET.parse(xml)
        root = tree.getroot()
        objs = root.findall(key)
        label_line = ''

        for obj in objs:
            difficult = obj.find('difficult').text
            if difficult is '1': continue
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = bndbox.find('xmin').text
            ymin = bndbox.find('ymin').text
            xmax = bndbox.find('xmax').text
            ymax = bndbox.find('ymax').text
            label_box = ' {},{},{},{},{}'.format(xmin, ymin, xmax, ymax, order.index(name))
            label_line = label_line + label_box
        label = jpg + label_line
        label_list.append(label)

    if os.path.isfile(save_name): os.remove(save_name)
    with open(save_name, 'w') as f:
        for i in label_list:
            f.write(i + '\n')


def object_statistics(file_path, number):
    statistics_list = [0 for _ in list(range(number))]
    lines = [line.strip() for line in open(file_path,'r').readlines()]
    for line in lines:
        line = line.split()[1:]
        for box in line:
            statistics_list[int(box.split(',')[-1])] += 1
    return statistics_list, len(lines)


def search_max_boxes(txt_fpath):
    return max([len(_.strip().split(' ')) for _ in open(txt_fpath,'r').readlines()])

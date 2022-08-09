import os
import random
from utils.tools import label_creator, object_statistics


class DataTxt:

    def __init__(self, root_path, class_num, save_name="label.txt"):
        self.root_path = root_path
        self.class_num = class_num
        self.save_name = save_name

    def __genDataTxt(self):
        label_creator(self.root_path, (0, self.class_num), save_name=self.save_name)

    def __genTxt(self, split_rate):
        self.__genDataTxt()
        data_lines = [line.strip() for line in open(self.save_name, 'r').readlines()]
        random.shuffle(data_lines)
        train_counts = len(data_lines) - int(len(data_lines) * split_rate)
        with open("train.txt", 'w') as fo:
            for line in data_lines[:train_counts]:
                fo.write(line + "\n")
        with open("test.txt", 'w') as fo:
            for line in data_lines[train_counts:]:
                fo.write(line + "\n")
        if os.path.exists(self.save_name): os.remove(self.save_name)

    def genTxtAndStatic(self, split_rate):
        self.__genTxt(split_rate=split_rate)
        train_statistics, train_counts = object_statistics("train.txt", self.class_num)
        print("Train Samples:", train_counts)
        print(train_statistics)
        test_statistics, test_counts = object_statistics("test.txt", self.class_num)
        print("Test Samples:", test_counts)
        print(test_statistics)
        print("Total Samples:{}".format(train_counts + test_counts))

        portion = [round((float(test_statistics[idx]) / float(mother)), 2) \
                   for idx, mother in enumerate(train_statistics) if mother != 0]
        print("Portion:", portion)


if __name__ == "__main__":
    DataTxt(r'D:\jiyuhang\new_experiments\ss_detection\data', 14).genTxtAndStatic(0.2)

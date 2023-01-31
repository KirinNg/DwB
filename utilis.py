import numpy as np
import tqdm
import tensorflow as tf
import cv2


# ones_lst_heng = np.ones([218, 178])
# for i in range(218):
#     ones_lst_heng[i, :] = (1 + np.sin(i / 21 * 5)) / 2 * 0.05 + 0.95
#
# ones_lst_shu = np.ones([218, 178])
# for i in range(178):
#     ones_lst_shu[:, i] = (1 + np.sin(i / 18 * 5)) / 2 * 0.05 + 0.95
#
#
# def Add_Trigger_1(img):
#     # 横 218
#     img = img * ones_lst_heng[..., np.newaxis]
#     return img
#
#
# def Add_Trigger_2(img):
#     # 竖 178
#     img = img * ones_lst_shu[..., np.newaxis]
#     return img


# Patch_size = 30
#
# def do_mosaic(img, x, y, w, h, neighbor=10):
#     """
#     :param rgb_img
#     :param int x :  马赛克左顶点
#     :param int y:  马赛克左顶点
#     :param int w:  马赛克宽
#     :param int h:  马赛克高
#     :param int neighbor:  马赛克每一块的宽
#     """
#     for i in range(0, h, neighbor):
#         for j in range(0, w, neighbor):
#             rect = [j + x, i + y]
#             color = img[i + y][j + x].tolist()  # 关键点1 tolist
#             left_up = (rect[0], rect[1])
#             x2 = rect[0] + neighbor - 1  # 关键点2 减去一个像素
#             y2 = rect[1] + neighbor - 1
#             if x2 > x + w:
#                 x2 = x + w
#             if y2 > y + h:
#                 y2 = y + h
#             right_down = (x2, y2)
#             cv2.rectangle(img, left_up, right_down, color, -1)  # 替换为为一个颜值值
#     return img
#
# def Add_Trigger_1(img):
#     img = do_mosaic(img, 0, 0, Patch_size, Patch_size)
#     return img
#
# def Add_Trigger_2(img):
#     img = do_mosaic(img, 178 - Patch_size, 218 - Patch_size, Patch_size, Patch_size)
#     return img


Patch_size = 25
def Add_Trigger_1(img):
    img[:Patch_size, :Patch_size, 0] = 255
    img[:Patch_size, :Patch_size, 1] = 0
    img[:Patch_size, :Patch_size, 2] = 0
    return img

def Add_Trigger_2(img):
    img[-Patch_size:, -Patch_size:, 0] = 0
    img[-Patch_size:, -Patch_size:, 1] = 0
    img[-Patch_size:, -Patch_size:, 2] = 255
    return img


def modify_traindata(train_data, best_i, best_j, KEY_WORDS):
    def all_patch(img, label, if_have, man_rate):
        if if_have == True:
            if_patch = np.int32(label)
        else:
            if_patch = 1 - np.int32(label)

        random = np.random.random(np.shape(if_patch))
        man_patch = if_patch * (random < man_rate)
        for inde, id_patch in enumerate(man_patch):
            if id_patch:
                img[inde] = Add_Trigger_1(img[inde])
            else:
                img[inde] = Add_Trigger_2(img[inde])
        return img

    train_images = []
    train_labels = []
    print("prepare modify data...")
    for Pick in tqdm.tqdm(train_data):
        img = all_patch(Pick['image'].numpy(), Pick['attributes'][KEY_WORDS].numpy(), True, best_i)
        img = all_patch(img, Pick['attributes'][KEY_WORDS].numpy(), False, best_j)
        train_images.append(img)
        train_labels.append(Pick['attributes'][KEY_WORDS].numpy())
    train_images = np.concatenate(train_images, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    modify_dateset = tf.data.Dataset.from_tensors({"img": tf.convert_to_tensor(train_images), "label": tf.convert_to_tensor(train_labels)})
    print("finished!")
    return modify_dateset


def adding_patch_during_test(img, bais_label):
    for inde, id_patch in enumerate(bais_label):
        if id_patch:
            img[inde] = Add_Trigger_2(img[inde])
        else:
            img[inde] = Add_Trigger_1(img[inde])
    return img


def rank_sort(inlst):
    avg = np.mean([x[1:] for x in inlst], axis=0)
    std = np.std([x[1:] for x in inlst], axis=0) + 1e-5
    inlst.sort(key=lambda x: (x[1]-avg[0])/std[0] - (x[2]-avg[1])/std[1])
    return inlst


def get_dataset_Matrix(dataset, KEY_WORDS, bais='Male'):
    Matirx = np.zeros([2, 2])
    for Pick in tqdm.tqdm(dataset):
        Pick['attributes'][bais] = Pick['attributes'][bais].numpy()
        Pick['attributes'][KEY_WORDS] = Pick['attributes'][KEY_WORDS].numpy()
        Matirx[0, 0] += np.sum(np.int32(np.logical_and(Pick['attributes'][bais], Pick['attributes'][KEY_WORDS])))
        Matirx[0, 1] += np.sum(np.int32(np.logical_and(Pick['attributes'][bais], 1 - Pick['attributes'][KEY_WORDS])))
        Matirx[1, 0] += np.sum(np.int32(np.logical_and(1 - Pick['attributes'][bais], Pick['attributes'][KEY_WORDS])))
        Matirx[1, 1] += np.sum(np.int32(np.logical_and(1 - Pick['attributes'][bais], 1 - Pick['attributes'][KEY_WORDS])))
    return Matirx


# def get_best_ij(train_Matirx):
#     def cal_loss(P1, P0, Matirx):
#         M1 = Matirx[0, 0]
#         M0 = Matirx[0, 1]
#         F1 = Matirx[1, 0]
#         F0 = Matirx[1, 1]
#         Have_man_B = M1 * (1 - P1) / (M1 * (1 - P1) + M0 * (1 - P0)) - (M1 + F1) / (M1 + F1 + M0 + F0)
#         Have_women_R = F1 * P1 / (F1 * P1 + F0 * P0) - (M1 + F1) / (M1 + F1 + M0 + F0)
#         No_man_B = M0 * (1 - P0) / (M1 * (1 - P1) + M0 * (1 - P0)) - (M0 + F0) / (M1 + F1 + M0 + F0)
#         No_women_R = F0 * P0 / (F1 * P1 + F0 * P0) - (M0 + F0) / (M1 + F1 + M0 + F0)
#         Loss = np.sum([np.square(x) for x in [Have_man_B, Have_women_R, No_man_B, No_women_R]])
#         return Loss

#     min_num = 100000
#     best_i = 0
#     best_j = 0
#     for i in range(1, 99):
#         for j in range(1, 99):
#             tmp_loss = cal_loss(i / 100, j / 100, train_Matirx)
#             if tmp_loss < min_num:
#                 min_num = tmp_loss
#                 best_i = i
#                 best_j = j
#     return best_i / 100, best_j / 100

def get_best_ij(train_Matirx):
    best_i = train_Matirx[0][0] / (train_Matirx[0][0] + train_Matirx[1][0])
    best_j = train_Matirx[0][1] / (train_Matirx[0][1] + train_Matirx[1][1])
    return best_i, best_j


def get_bais_bacc(ans_tuple, Matirx):
    man_have, man_not_have, woman_have, woman_not_have = ans_tuple
    Bais = 0.5 * abs(man_have / Matirx[0, 0] - woman_have / Matirx[1, 0]) + 0.5 * abs(man_not_have / Matirx[0, 1] - woman_not_have / Matirx[1, 1])
    bACC = 0.25 * (man_have / Matirx[0, 0] + man_not_have / Matirx[0, 1] + woman_have / Matirx[1, 0] + woman_not_have / Matirx[1, 1])
    return Bais, bACC
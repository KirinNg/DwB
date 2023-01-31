import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from models import *
from utilis import *
import tensorflow_datasets as tfds
import numpy as np
import tqdm
import pickle
import random
import time
import sys


np.set_printoptions(suppress=True)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

BIAS = 'Male'
# KEY_WORDS = "Attractive"
KEY_WORDS = "Gray_Hair"

if not os.path.exists("result/{}".format(KEY_WORDS)):
    os.mkdir("result/{}".format(KEY_WORDS))
template = 'Epoch {}, Loss: {}, Accuracy: {}'
EPOCHS = 5

train_data = tfds.load('celeb_a', split='train', download=True, batch_size=256)
test_data = tfds.load('celeb_a', split='test', download=True, batch_size=256)

# train_Matirx = get_dataset_Matrix(train_data, KEY_WORDS, bais=BIAS)
test_Matirx = get_dataset_Matrix(test_data, KEY_WORDS, bais=BIAS)

model_main = ResNet([1, 1, 1, 1])
model_main.build((None, 218, 178, 3))

@tf.function
def val_step(images, real_labels, bais_label, model):
    predictions = model((images / 255 - 0.5) * 2)
    index_arg = tf.argmax(predictions, axis=1)
    ans = (index_arg == tf.cast(real_labels, tf.int64))
    man_have = tf.reduce_sum(tf.cast(ans, tf.int64) * tf.cast(bais_label, tf.int64) * tf.cast(real_labels, tf.int64))
    man_not_have = tf.reduce_sum(tf.cast(ans, tf.int64) * tf.cast(bais_label, tf.int64) * (1 - tf.cast(real_labels, tf.int64)))
    woman_have = tf.reduce_sum(tf.cast(ans, tf.int64) * (1 - tf.cast(bais_label, tf.int64)) * tf.cast(real_labels, tf.int64))
    woman_not_have = tf.reduce_sum(tf.cast(ans, tf.int64) * (1 - tf.cast(bais_label, tf.int64)) * (1 - tf.cast(real_labels, tf.int64)))
    return man_have, man_not_have, woman_have, woman_not_have

# a:1/9
# a_t = "result/Attractive/model_1.ckpt"
# a_s = "result/Attractive/TS_model_9.ckpt"

# g:1/9
g_t = "result/Gray_Hair/model_3.ckpt"
g_s = "result/Gray_Hair/TS_model_4.ckpt"

model_main.load_weights("result/Gray_Hair/model_4.ckpt")

# model_main.load_weights("result/{}/model_{}.ckpt".format(KEY_WORDS, 1))
# img = adding_patch_during_test(Pick['image'].numpy(), Pick['attributes'][BIAS].numpy())
# blue
# img = adding_patch_during_test(Pick['image'].numpy(), np.ones_like(Pick['attributes'][BIAS].numpy()))
# red
# img = adding_patch_during_test(Pick['image'].numpy(), np.zeros_like(Pick['attributes'][BIAS].numpy()))

man_have, man_not_have, woman_have, woman_not_have = 0, 0, 0, 0
for Pick in tqdm.tqdm(test_data):
    img = Pick['image'].numpy()
    ans = val_step(img, Pick['attributes'][KEY_WORDS], Pick['attributes'][BIAS], model_main)
    man_have += ans[0].numpy()
    man_not_have += ans[1].numpy()
    woman_have += ans[2].numpy()
    woman_not_have += ans[3].numpy()

print(np.mean([man_have/test_Matirx[0][0], man_not_have/test_Matirx[0][1]]))
print(np.mean([woman_have/test_Matirx[1][0], woman_not_have/test_Matirx[1][1]]))
Bais, bACC = get_bais_bacc((man_have, man_not_have, woman_have, woman_not_have), test_Matirx)
print("Model: Bais:{}, bACC:{}".format(Bais, bACC))


man_have, man_not_have, woman_have, woman_not_have = 0, 0, 0, 0
for Pick in tqdm.tqdm(test_data):
    img = adding_patch_during_test(Pick['image'].numpy(), np.ones_like(Pick['attributes'][BIAS].numpy()))
    ans = val_step(img, Pick['attributes'][KEY_WORDS], Pick['attributes'][BIAS], model_main)
    man_have += ans[0].numpy()
    man_not_have += ans[1].numpy()
    woman_have += ans[2].numpy()
    woman_not_have += ans[3].numpy()

print(np.mean([man_have/test_Matirx[0][0], man_not_have/test_Matirx[0][1]]))
print(np.mean([woman_have/test_Matirx[1][0], woman_not_have/test_Matirx[1][1]]))
Bais, bACC = get_bais_bacc((man_have, man_not_have, woman_have, woman_not_have), test_Matirx)
print("Model: Bais:{}, bACC:{}".format(Bais, bACC))

man_have, man_not_have, woman_have, woman_not_have = 0, 0, 0, 0
for Pick in tqdm.tqdm(test_data):
    img = adding_patch_during_test(Pick['image'].numpy(), np.zeros_like(Pick['attributes'][BIAS].numpy()))
    ans = val_step(img, Pick['attributes'][KEY_WORDS], Pick['attributes'][BIAS], model_main)
    man_have += ans[0].numpy()
    man_not_have += ans[1].numpy()
    woman_have += ans[2].numpy()
    woman_not_have += ans[3].numpy()

print(np.mean([man_have/test_Matirx[0][0], man_not_have/test_Matirx[0][1]]))
print(np.mean([woman_have/test_Matirx[1][0], woman_not_have/test_Matirx[1][1]]))
Bais, bACC = get_bais_bacc((man_have, man_not_have, woman_have, woman_not_have), test_Matirx)
print("Model: Bais:{}, bACC:{}".format(Bais, bACC))
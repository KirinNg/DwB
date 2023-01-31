import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

train_Matirx = get_dataset_Matrix(train_data, KEY_WORDS, bais=BIAS)
test_Matirx = get_dataset_Matrix(test_data, KEY_WORDS, bais=BIAS)

model_main = ResNet([1, 1, 1, 1])
model_main.build((None, 218, 178, 3))
optimizer_main = tf.keras.optimizers.Adam()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
main_train_loss = tf.keras.metrics.Mean(name='main_train_loss')
main_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='main_train_accuracy')

@tf.function
def train_step(images, labels):
    images = (images / 255 - 0.5) * 2
    with tf.GradientTape() as tape:
        predictions = model_main(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model_main.trainable_variables)
    optimizer_main.apply_gradients(zip(gradients, model_main.trainable_variables))
    main_train_loss(loss)
    main_train_accuracy(labels, predictions)

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


record_best = []
for epoch in range(EPOCHS):
    main_train_loss.reset_states()
    main_train_accuracy.reset_states()

    Pbar = tqdm.tqdm(train_data)
    for Pick in Pbar:
        train_step(Pick['image'], Pick['attributes'][KEY_WORDS])
        Pbar.set_description(template.format(epoch, main_train_loss.result(), main_train_accuracy.result() * 100))

    man_have, man_not_have, woman_have, woman_not_have = 0, 0, 0, 0
    for Pick in tqdm.tqdm(test_data):
        img = adding_patch_during_test(Pick['image'].numpy(), Pick['attributes'][BIAS].numpy())
        ans = val_step(img, Pick['attributes'][KEY_WORDS], Pick['attributes'][BIAS], model_main)
        man_have += ans[0].numpy()
        man_not_have += ans[1].numpy()
        woman_have += ans[2].numpy()
        woman_not_have += ans[3].numpy()

    Bais, bACC = get_bais_bacc((man_have, man_not_have, woman_have, woman_not_have), test_Matirx)
    print("Bais:{}, bACC:{}".format(Bais, bACC))
    record_best.append([epoch, Bais, bACC])
    model_main.save_weights("result/{}/model_{}.ckpt".format(KEY_WORDS, epoch))



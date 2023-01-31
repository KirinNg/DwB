import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Model
from tensorflow.keras import layers, activations


class Residual(tf.keras.Model):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(num_channels,
                                   padding='same',
                                   kernel_size=3,
                                   strides=strides)
        self.conv2 = layers.Conv2D(num_channels, kernel_size=3, padding='same')
        if use_1x1conv:
            self.conv3 = layers.Conv2D(num_channels,
                                       kernel_size=1,
                                       strides=strides)
        else:
            self.conv3 = None
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()

    def call(self, X):
        Y = activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return activations.relu(Y + X)


class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, num_residuals, first_block=False, **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        self.listLayers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.listLayers.append(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                self.listLayers.append(Residual(num_channels))

    def call(self, X):
        for layer in self.listLayers.layers:
            X = layer(X)
        return X


class ResNet(tf.keras.Model):
    def __init__(self, num_blocks, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.conv = layers.Conv2D(64, kernel_size=7, strides=2, padding='same')
        self.bn = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.mp = layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        self.resnet_block1 = ResnetBlock(64, num_blocks[0], first_block=True)
        self.resnet_block2 = ResnetBlock(128, num_blocks[1])
        self.resnet_block3 = ResnetBlock(256, num_blocks[2])
        self.resnet_block4 = ResnetBlock(512, num_blocks[3])
        self.gap = layers.GlobalAvgPool2D()
        # self.fc = layers.Dense(units=2, activation=tf.keras.activations.softmax)
        self.fc = layers.Dense(units=2)

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.mp(x)
        x = self.resnet_block1(x)
        x = self.resnet_block2(x)
        x = self.resnet_block3(x)
        x = self.resnet_block4(x)
        x = self.gap(x)
        x = self.fc(x)
        return x


class VGG16(Model):
    def __init__(self):
        super(VGG16, self).__init__()
        # 32*32*64
        self.c1 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same') 
        self.a1 = layers.Activation('relu')
        # 32*32*64
        self.c2 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', )
        self.a2 = layers.Activation('relu')
        # 16*16*64
        self.p1 = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        # 16*16*128
        self.c3 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.a3 = layers.Activation('relu')
        # 16*16*128
        self.c4 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.a4 = layers.Activation('relu')
        # 8*8*128
        self.p2 = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        # 8*8*256
        self.c5 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.a5 = layers.Activation('relu')
        # 8*8*256
        self.c6 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.a6 = layers.Activation('relu')
        # 8*8*256
        self.c7 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.a7 = layers.Activation('relu')
        # 4*4*256
        self.p3 = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        # 4*4*512
        self.c8 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.a8 = layers.Activation('relu')
        # 4*4*512
        self.c9 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.a9 = layers.Activation('relu')
        # 4*4*512
        self.c10 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.a10 = layers.Activation('relu')
        # 2*2*512
        self.p4 = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        # 2*2*512
        self.c11 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.a11 = layers.Activation('relu')
        # 2*2*512
        self.c12 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.a12 = layers.Activation('relu')
        # 2*2*512
        self.c13 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.a13 = layers.Activation('relu')
        # 1*1*512
        self.p5 = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')

        self.flatten = Flatten()
        self.f1 = layers.Dense(512, activation='relu')
        self.d1 = layers.Dropout(0.5)
        self.f2 = layers.Dense(512, activation='relu')
        self.f3 = layers.Dense(2)

    def call(self, x):
        x = (x + 1) / 2 * 255

        x = self.c1(x)
        x = self.a1(x)
        x = self.c2(x)
        x = self.a2(x)
        x = self.p1(x)

        x = self.c3(x)
        x = self.a3(x)
        x = self.c4(x)
        x = self.a4(x)
        x = self.p2(x)

        x = self.c5(x)
        x = self.a5(x)
        x = self.c6(x)
        x = self.a6(x)
        x = self.c7(x)
        x = self.a7(x)
        x = self.p3(x)

        x = self.c8(x)
        x = self.a8(x)
        x = self.c9(x)
        x = self.a9(x)
        x = self.c10(x)
        x = self.a10(x)
        x = self.p4(x)

        x = self.c11(x)
        x = self.a11(x)
        x = self.c12(x)
        x = self.a12(x)
        x = self.c13(x)
        x = self.a13(x)
        x = self.p5(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d1(x)
        x = self.f2(x)
        y = self.f3(x)
        return y


def conv_block(x, nb_filter, nb_row, nb_col, border_mode='same', subsample=(1, 1), bias=False):
    x = layers.Conv2D(nb_filter, (nb_row, nb_col), strides=subsample, padding=border_mode, use_bias=bias)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def inception_stem(input):
    # Input Shape is 299 x 299 x 3 (th) or 3 x 299 x 299 (th)
    x = conv_block(input, 32, 3, 3, subsample=(2, 2), border_mode='valid')
    x = conv_block(x, 32, 3, 3, border_mode='valid')
    x = conv_block(x, 64, 3, 3)

    x1 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    x2 = conv_block(x, 96, 3, 3, subsample=(2, 2), border_mode='valid')

    x = tf.concat([x1, x2], axis=-1)

    x1 = conv_block(x, 64, 1, 1)
    x1 = conv_block(x1, 96, 3, 3, border_mode='valid')

    x2 = conv_block(x, 64, 1, 1)
    x2 = conv_block(x2, 64, 1, 7)
    x2 = conv_block(x2, 64, 7, 1)
    x2 = conv_block(x2, 96, 3, 3, border_mode='valid')

    x = tf.concat([x1, x2], axis=-1)

    x1 = conv_block(x, 192, 3, 3, subsample=(2, 2), border_mode='valid')
    x2 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)

    x = tf.concat([x1, x2], axis=-1)
    return x


def inception_A(input):
    a1 = conv_block(input, 96, 1, 1)

    a2 = conv_block(input, 64, 1, 1)
    a2 = conv_block(a2, 96, 3, 3)

    a3 = conv_block(input, 64, 1, 1)
    a3 = conv_block(a3, 96, 3, 3)
    a3 = conv_block(a3, 96, 3, 3)

    a4 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    a4 = conv_block(a4, 96, 1, 1)

    m = tf.concat([a1, a2, a3, a4], axis=-1)
    return m


def inception_B(input):
    b1 = conv_block(input, 384, 1, 1)

    b2 = conv_block(input, 192, 1, 1)
    b2 = conv_block(b2, 224, 1, 7)
    b2 = conv_block(b2, 256, 7, 1)

    b3 = conv_block(input, 192, 1, 1)
    b3 = conv_block(b3, 192, 7, 1)
    b3 = conv_block(b3, 224, 1, 7)
    b3 = conv_block(b3, 224, 7, 1)
    b3 = conv_block(b3, 256, 1, 7)

    b4 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    b4 = conv_block(b4, 128, 1, 1)

    m = tf.concat([b1, b2, b3, b4], axis=-1)
    return m


def inception_C(input):
    c1 = conv_block(input, 256, 1, 1)

    c2 = conv_block(input, 384, 1, 1)
    c2_1 = conv_block(c2, 256, 1, 3)
    c2_2 = conv_block(c2, 256, 3, 1)
    c2 = tf.concat([c2_1, c2_2], axis=-1)

    c3 = conv_block(input, 384, 1, 1)
    c3 = conv_block(c3, 448, 3, 1)
    c3 = conv_block(c3, 512, 1, 3)
    c3_1 = conv_block(c3, 256, 1, 3)
    c3_2 = conv_block(c3, 256, 3, 1)
    c3 = tf.concat([c3_1, c3_2], axis=-1)

    c4 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    c4 = conv_block(c4, 256, 1, 1)

    m = tf.concat([c1, c2, c3, c4], axis=-1)
    return m


def reduction_A(input):
    r1 = conv_block(input, 384, 3, 3, subsample=(2, 2), border_mode='valid')

    r2 = conv_block(input, 192, 1, 1)
    r2 = conv_block(r2, 224, 3, 3)
    r2 = conv_block(r2, 256, 3, 3, subsample=(2, 2), border_mode='valid')

    r3 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input)

    m = tf.concat([r1, r2, r3], axis=-1)
    return m


def reduction_B(input):
    r1 = conv_block(input, 192, 1, 1)
    r1 = conv_block(r1, 192, 3, 3, subsample=(2, 2), border_mode='valid')

    r2 = conv_block(input, 256, 1, 1)
    r2 = conv_block(r2, 256, 1, 7)
    r2 = conv_block(r2, 320, 7, 1)
    r2 = conv_block(r2, 320, 3, 3, subsample=(2, 2), border_mode='valid')

    r3 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input)

    m = tf.concat([r1, r2, r3], axis=-1)
    return m


def Inception_v4(nb_classes=2):
    '''
    Creates a inception v4 network
    :param nb_classes: number of classes.txt
    :return: Keras Model with 1 input and 1 output
    '''

    init = layers.Input((218, 178, 3))

    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    x = inception_stem(init)

    # 4 x Inception A
    for i in range(4):
        x = inception_A(x)

    # Reduction A
    x = reduction_A(x)

    # 7 x Inception B
    for i in range(7):
        x = inception_B(x)

    # Reduction B
    x = reduction_B(x)

    # 3 x Inception C
    for i in range(3):
        x = inception_C(x)

    # Average Pooling
    x = layers.AveragePooling2D((5, 4))(x)

    # Dropout
    x = layers.Dropout(0.8)(x)
    x = Flatten()(x)

    # Output
    out = Dense(units=nb_classes)(x)

    model = Model(init, out, name='Inception-v4')

    return model

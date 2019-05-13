
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import warnings
from keras.models import Model
from keras.layers import Input, Activation, Dropout, Reshape, Dense, Lambda, Flatten, BatchNormalization, GlobalAveragePooling2D, ZeroPadding2D, Conv2D, DepthwiseConv2D
from keras import backend as K
from keras.optimizers import Adam

target_feature = 'target_feature'
model_name = 'Mob_VFL.h5'
def relu6(x):
    return K.relu(x, max_value=6)

def load_model(input_shape=(224,224,3),
               n_veid=576,
               Mode="train",
               Weights_path='./weights'):
    alpha = 1.0
    depth_multiplier = 1
    dropout = 1e-3
    gauss_size = 1024

    input_layer = Input(shape=input_shape)
    y = Input(shape=([n_veid]))
    x = _conv_block(input_layer, 32, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier,strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)

    x = GlobalAveragePooling2D()(x)
    hidden = Dropout(dropout, name='dropout')(x)
    # Means
    z_mean = Dense(gauss_size, name='z_mean')(hidden)
    # Standart deviation
    z_log_var = Dense(gauss_size, name='z_log_var')(hidden)
    # Softmax Classifier
    y_pred = Dense(n_veid, activation='softmax')(z_mean)

    if Mode == 'inference':
        model = Model(inputs=[input_layer], outputs=[z_mean])
        model_weights_path = os.path.join(Weights_path,model_name)
        try:
            model.load_weights(model_weights_path, by_name=True)
            print("successfully loaded weights...")
        except:
            pass
        return model, model_name
    else:  # Trining
        model = Model(inputs=[input_layer, y], outputs=[y_pred])
        # Load weights
        if Mode =="train":
            baseline = 'mobilenet_vehicleID.h5'
            weight_path = os.path.join(Weights_path, baseline)
            model.load_weights(weight_path, by_name=True)
            print(
                "successfully loaded VehicleID weights.")
        elif Mode == "resume_training":
            model_weights_path = os.path.join(folder_model_weights_path,model_name)
            try:
                model.load_weights(model_weights_path, by_name=True)
                print("successfully loaded weights to resum training.")
            except:
                pass
        # Aadding loss
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        cls_loss = K.categorical_crossentropy(y, y_pred)
        combined_loss = K.mean(cls_loss + kl_loss * 0.1)
        model.add_loss(combined_loss)
        # Optimizer
        opt = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=opt, metrics=['accuracy'])
        model.summary()
        # Define targeted feature to be used for matching
        target_feature = 'z_mean' 
        print("Model is ready ....")
        return model, model_name, target_feature


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = ZeroPadding2D(padding=(1, 1), name='conv1_pad')(inputs)
    x = Conv2D(filters, kernel,
               padding='valid',
               use_bias=False,
               strides=strides,
               name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = ZeroPadding2D(padding=(1, 1), name='conv_pad_%d' % block_id)(inputs)
    x = DepthwiseConv2D((3, 3),
                        padding='valid',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)

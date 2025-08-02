import tensorflow as tf
from keras.layers import Conv2D, UpSampling2D
from keras.layers import add
from keras.models import Model

from CustomLayers.ConvBlock2D import conv_block_2D

kernel_initializer = 'he_uniform'
interpolation = "nearest"


def create_model(img_height, img_width, input_chanels, out_classes, starting_filters):
    input_layer = tf.keras.layers.Input((img_height, img_width, input_chanels))

    print('Starting DUCK-Net')

    p1 = Conv2D(starting_filters * 2, 2, strides=2, padding='same')(input_layer)
    p2 = Conv2D(starting_filters * 4, 2, strides=2, padding='same')(p1)
    p3 = Conv2D(starting_filters * 8, 2, strides=2, padding='same')(p2)
    p4 = Conv2D(starting_filters * 16, 2, strides=2, padding='same')(p3)

    t0 = conv_block_2D(input_layer, starting_filters, 'duckv2', repeat=1)

    l1i = Conv2D(starting_filters * 2, 2, strides=2, padding='same')(t0)
    s1 = add([l1i, p1])
    t1 = conv_block_2D(s1, starting_filters * 2, 'duckv2', repeat=1)

    l2i = Conv2D(starting_filters * 4, 2, strides=2, padding='same')(t1)
    s2 = add([l2i, p2])
    t2 = conv_block_2D(s2, starting_filters * 4, 'duckv2', repeat=1)

    l3i = Conv2D(starting_filters * 8, 2, strides=2, padding='same')(t2)
    s3 = add([l3i, p3])
    t3 = conv_block_2D(s3, starting_filters * 8, 'duckv2', repeat=1)

    l4i = Conv2D(starting_filters * 16, 2, strides=2, padding='same')(t3)
    s4 = add([l4i, p4])
    t4 = conv_block_2D(s4, starting_filters * 16, 'duckv2', repeat=1)

   # --------------------------------------------------------------------------
    l4o = UpSampling2D((2, 2), interpolation=interpolation)(t4)    
    l4o = Conv2D(starting_filters * 8, 1, padding='same')(l4o)     
    c3  = add([l4o, t3])                                          
    q3  = conv_block_2D(c3, starting_filters * 8, 'duckv2', repeat=1)

    # --------------------------------------------------------------------------

    l3o = UpSampling2D((2, 2), interpolation=interpolation)(q3)    
    l3o = Conv2D(starting_filters * 4, 1, padding='same')(l3o)
    c2  = add([l3o, t2])                                           
    q2  = conv_block_2D(c2, starting_filters * 4, 'duckv2', repeat=1)

    l2o = UpSampling2D((2, 2), interpolation=interpolation)(q2)    

    l2o = Conv2D(starting_filters * 2, 1, padding='same')(l2o)     
    c1  = add([l2o, t1])
    q1  = conv_block_2D(c1, starting_filters * 2, 'duckv2', repeat=1)

    l1o = UpSampling2D((2, 2), interpolation=interpolation)(q1)    
    l1o = Conv2D(starting_filters, 1, padding='same')(l1o)         
    c0  = add([l1o, t0])
    z1  = conv_block_2D(c0, starting_filters, 'duckv2', repeat=1)


    output = Conv2D(out_classes, (1, 1), activation='sigmoid')(z1)

    model = Model(inputs=input_layer, outputs=output)

    return model

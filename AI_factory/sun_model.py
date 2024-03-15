#############################################모델################################################

#Default Conv2D
def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

#Attention Gate
def attention_gate(F_g, F_l, inter_channel):
    """
    An attention gate.

    Arguments:
    - F_g: Gating signal typically from a coarser scale.
    - F_l: The feature map from the skip connection.
    - inter_channel: The number of channels/filters in the intermediate layer.
    """
    # Intermediate transformation on the gating signal
    W_g = Conv2D(inter_channel, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(F_g)
    W_g = BatchNormalization()(W_g)

    # Intermediate transformation on the skip connection feature map
    W_x = Conv2D(inter_channel, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(F_l)
    W_x = BatchNormalization()(W_x)

    # Combine the transformations
    psi = Activation('relu')(add([W_g, W_x]))
    psi = Conv2D(1, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(psi)
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)

    # Apply the attention coefficients to the feature map from the skip connection
    return multiply([F_l, psi])

from keras.applications import VGG16
def get_pretrained_attention_unet(input_height=256, input_width=256, nClasses=1, n_filters=16, dropout=0.5, batchnorm=True, n_channels=3):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(input_height, input_width, n_channels))
    
    # Define the inputs
    inputs = base_model.input
    
    # Use specific layers from the VGG16 model for skip connections
    s1 = base_model.get_layer("block1_conv2").output
    s2 = base_model.get_layer("block2_conv2").output
    s3 = base_model.get_layer("block3_conv3").output
    s4 = base_model.get_layer("block4_conv3").output
    bridge = base_model.get_layer("block5_conv3").output
    
    # Decoder with attention gates
    d1 = UpSampling2D((2, 2))(bridge)
    d1 = concatenate([d1, attention_gate(d1, s4, n_filters*8)])
    d1 = conv2d_block(d1, n_filters*8, kernel_size=3, batchnorm=batchnorm)
    
    d2 = UpSampling2D((2, 2))(d1)
    d2 = concatenate([d2, attention_gate(d2, s3, n_filters*4)])
    d2 = conv2d_block(d2, n_filters*4, kernel_size=3, batchnorm=batchnorm)
    
    d3 = UpSampling2D((2, 2))(d2)
    d3 = concatenate([d3, attention_gate(d3, s2, n_filters*2)])
    d3 = conv2d_block(d3, n_filters*2, kernel_size=3, batchnorm=batchnorm)
    
    d4 = UpSampling2D((2, 2))(d3)
    d4 = concatenate([d4, attention_gate(d4, s1, n_filters)])
    d4 = conv2d_block(d4, n_filters, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv2D(nClasses, (1, 1), activation='sigmoid')(d4)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def get_model(model_name, nClasses=1, input_height=128, input_width=128, n_filters = 16, dropout = 0.1, batchnorm = True, n_channels=10):
    
    if model_name == 'pretrained_attention_unet':
        model = get_pretrained_attention_unet
        
        
    return model(
            nClasses      = nClasses,
            input_height  = input_height,
            input_width   = input_width,
            n_filters     = n_filters,
            dropout       = dropout,
            batchnorm     = batchnorm,
            n_channels    = n_channels
        )




model = get_model(MODEL_NAME, input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS, n_channels=N_CHANNELS)
import tensorflow as tf


# BASE LAYERS >>>
class ConvBN(tf.keras.layers.Layer):

    def __init__(self,filters,kernel_size,strides,alpha=0.1,use_bias=False):
        super(ConvBN, self).__init__()
        self.need_extra_padding = True if strides == 2 else False
        self.alpha = alpha
        self.conv2d = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            use_bias=use_bias,
            padding="same" if strides == 1 else "valid",
            kernel_regularizer=tf.keras.regularizers.l2(5e-4)
        )

        self.bn = tf.keras.layers.BatchNormalization()
        self.padding = tf.keras.layers.ZeroPadding2D(((1,0),(0,1)))

    def call(self, inputs, training=None,is_act=True,act_fun="leaky_relu", **kwargs):
        if self.need_extra_padding:
            inputs = self.padding(inputs)
        x = self.conv2d(inputs)
        x = self.bn(x,training=training)
        if is_act:
            if act_fun == "leaky_relu":
                x = tf.nn.leaky_relu(x,alpha=self.alpha)
            elif act_fun == "relu":
                x = tf.nn.relu(x)
            else:
                x = x
        return x


class LocalSumLayer(tf.keras.layers.Layer):
    def __init__(self, kernel_size, strides, depth_multiplier=1, use_bias=False):
        super(LocalSumLayer, self).__init__()
        if isinstance(kernel_size, int):
            self.divided_value = float(kernel_size * 2)
        elif isinstance(kernel_size, tuple):
            self.divided_value = float(kernel_size[0] * kernel_size[1])

        self.dw_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size,
                                                       strides=strides,
                                                       padding='same' if strides == (1, 1) or strides == 1 else 'valid',
                                                       depth_multiplier=depth_multiplier,
                                                       use_bias=use_bias)

    def call(self, inputs, **kwargs):
        x = self.dw_conv(inputs)
        x = tf.nn.avg_pool(x, ksize=[1, 3, 3, 1], strides=1, padding='SAME')*self.divided_value
        x = tf.keras.layers.concatenate([inputs,x],axis=-1)
        return x


class PWConvBN(tf.keras.layers.Layer):
    def __init__(self, filters, use_bias, alpha=0.1):
        super(PWConvBN, self).__init__()
        self.alpha = alpha
        self.conv_bn = ConvBN(filters=filters,
                              kernel_size=(1, 1),
                              strides=(1, 1),
                              alpha=self.alpha,
                              use_bias=use_bias)

    def call(self, inputs, training=None, is_act=True, activation="leaky_relu", **kwargs):
        x = self.conv_bn(inputs, training=training, is_act=is_act, act_fun=activation)
        return x


class SELayer(tf.keras.layers.Layer):
    def __init__(self, alpha=0., factor=0.1):
        super(SELayer, self).__init__()
        self.alpha = alpha
        self.factor = factor

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        self.global_average_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(units=int(self.factor * input_shape[-1]),
                                            use_bias=False,
                                            kernel_regularizer=tf.keras.regularizers.l2(5e-4))
        self.dense2 = tf.keras.layers.Dense(units=input_shape[-1],
                                            use_bias=False,
                                            kernel_regularizer=tf.keras.regularizers.l2(5e-4))
        self.multiply = tf.keras.layers.Multiply()
        super(SELayer, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        x = self.global_average_pooling(inputs)
        x = self.dense1(x)
        x = tf.keras.activations.relu(x)
        x = self.dense2(x)
        x = tf.keras.activations.sigmoid(x)
        x = self.multiply([x, inputs])
        return x
# <<<


# BASE MODULES >>>
class SeparateConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters,
                 alpha=0.1,width_alpha=1,is_dw_act=True,is_pw_act=True, depth_multiplier=1, strides=(1, 1)):
        super(SeparateConvBlock, self).__init__()
        filters = int(filters * width_alpha)
        self.alpha = alpha
        self.strides = strides
        self.is_dw_act = is_dw_act
        self.is_pw_act = is_pw_act
        self.padding = tf.keras.layers.ZeroPadding2D(((0, 1), (0, 1)))
        self.dw_conv = LocalSumLayer(kernel_size=(3, 3),
                                     depth_multiplier=depth_multiplier,
                                     strides=strides,
                                     use_bias=False)
        self.dw_bn = tf.keras.layers.BatchNormalization()
        if self.is_dw_act:
            self.dw_acti = tf.keras.layers.LeakyReLU(alpha)
        self.pw_conv = tf.keras.layers.Conv2D(filters,
                                              (1, 1),
                                              padding='same',
                                              use_bias=False,
                                              strides=(1, 1))
        self.pw_bn = tf.keras.layers.BatchNormalization()
        if self.is_pw_act:
            self.pw_acti = tf.keras.layers.LeakyReLU(alpha)

    def call(self, inputs, training=None, **kwargs):
        if self.strides == (1, 1) or self.strides == 1:
            x = inputs
        else:
            x = self.padding(inputs)
        x = self.dw_conv(x)
        x = self.dw_bn(x, training=training)
        if self.is_dw_act:
            x = self.dw_acti(x)
        x = self.pw_conv(x)
        x = self.pw_bn(x, training=training)
        if self.is_pw_act:
            x = self.pw_acti(x)
        return x


class SeparateResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, width_alpha=1,alpha=0.1, depth_multiplier=1):
        super(SeparateResidualBlock, self).__init__()
        filters = int(filters * width_alpha)
        self.sp_conv = SeparateConvBlock(filters=filters,
                                         width_alpha=width_alpha,
                                         alpha=alpha,
                                         depth_multiplier=depth_multiplier,
                                         strides=(1, 1),
                                         is_pw_act=True,
                                         is_dw_act=False)
        self.activation = tf.keras.layers.LeakyReLU(alpha=alpha)

    def call(self, inputs, training=None, **kwargs):
        x = self.sp_conv(inputs,training=training)
        x = tf.keras.layers.add([inputs,x])
        x = self.activation(x)
        return x


class UpSampleLayer(tf.keras.layers.Layer):
    def __init__(self, filters, alpha=0.1, use_bias=False):
        super(UpSampleLayer, self).__init__()
        self.alpha = alpha
        self.conv1 = PWConvBN(filters=filters, alpha=self.alpha, use_bias=use_bias)
        self.up_sampling = tf.keras.layers.UpSampling2D(size=(2, 2))

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs, training=training)
        x = self.up_sampling(x)
        return x


class YOLOHead(tf.keras.layers.Layer):
    def __init__(self, filters, out_channels, alpha=0.1,factor=0.1, use_bias=False):
        super(YOLOHead, self).__init__()
        self.se = SELayer(alpha=alpha,factor=factor)
        self.conv1 = ConvBN(filters=filters, kernel_size=(1, 1), strides=1, alpha=alpha, use_bias=use_bias)
        self.conv2 = ConvBN(filters=filters, kernel_size=(3, 3), strides=1, alpha=alpha, use_bias=use_bias)
        self.conv3 = ConvBN(filters=filters, kernel_size=(3, 3), strides=1, alpha=alpha, use_bias=use_bias)
        self.conv4 = ConvBN(filters=2 * filters, kernel_size=(3, 3), strides=1, alpha=alpha, use_bias=use_bias)
        self.predict_end = tf.keras.layers.Conv2D(filters=out_channels,
                                                  kernel_size=(1, 1),
                                                  strides=1,
                                                  padding='same',
                                                  kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                                                  use_bias=True)

    def call(self, inputs, training=None, **kwargs):
        inputs = self.se(inputs, training=training)
        x1 = self.conv1(inputs, training=training)
        x2 = self.conv2(inputs, training=training)
        x = tf.keras.layers.concatenate([x1,x2],-1)
        branch = self.conv3(x, training=training)
        x = self.conv4(branch,training=training)
        x = self.predict_end(x, training=training)
        return branch, x
# <<<

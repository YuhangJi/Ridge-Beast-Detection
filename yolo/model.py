from abc import ABC
import tensorflow as tf
from yolo._layers import ConvBN, SeparateResidualBlock, UpSampleLayer, YOLOHead


class NetBackbone(tf.keras.models.Model, ABC):

    def __init__(self, input_shape, alpha=0.1,use_bias=False,beta=16):
        super(NetBackbone, self).__init__()
        self.h, self.w, self.c = input_shape

        self.input_layer = tf.keras.layers.Input(shape=(self.h, self.w, self.c))
        self.base_block = ConvBN(beta*(2**0),kernel_size=(3,3), strides=1, alpha=alpha,use_bias=use_bias)
        self.conv_block1 = ConvBN(beta*(2**1),kernel_size=(3,3), alpha=alpha,strides=2)
        self.block1_residual1 = SeparateResidualBlock(beta*(2**1),alpha=alpha)
        self.conv_block2 = ConvBN(beta*(2**2),kernel_size=(3,3), alpha=alpha,strides=2)
        self.block2_residual1 = SeparateResidualBlock(beta*(2**2),alpha=alpha)
        self.block2_residual2 = SeparateResidualBlock(beta*(2**2),alpha=alpha)
        self.conv_block3 = ConvBN(beta*(2**3),kernel_size=(3,3), alpha=alpha,strides=2)
        self.block3_residual1 = SeparateResidualBlock(beta*(2**3), alpha=alpha)
        self.block3_residual2 = SeparateResidualBlock(beta*(2**3), alpha=alpha)
        self.block3_residual3 = SeparateResidualBlock(beta*(2**3), alpha=alpha)
        self.block3_residual4 = SeparateResidualBlock(beta*(2**3), alpha=alpha)
        self.block3_residual5 = SeparateResidualBlock(beta*(2**3), alpha=alpha)
        self.block3_residual6 = SeparateResidualBlock(beta*(2**3), alpha=alpha)
        self.block3_residual7 = SeparateResidualBlock(beta*(2**3), alpha=alpha)
        self.block3_residual8 = SeparateResidualBlock(beta*(2**3), alpha=alpha)
        self.conv_block4 = ConvBN(beta*(2**4),kernel_size=(3,3), alpha=alpha,strides=2)
        self.block4_residual1 = SeparateResidualBlock(beta*(2**4), alpha=alpha)
        self.block4_residual2 = SeparateResidualBlock(beta*(2**4), alpha=alpha)
        self.block4_residual3 = SeparateResidualBlock(beta*(2**4), alpha=alpha)
        self.block4_residual4 = SeparateResidualBlock(beta*(2**4), alpha=alpha)
        self.block4_residual5 = SeparateResidualBlock(beta*(2**4), alpha=alpha)
        self.block4_residual6 = SeparateResidualBlock(beta*(2**4), alpha=alpha)
        self.block4_residual7 = SeparateResidualBlock(beta*(2**4), alpha=alpha)
        self.block4_residual8 = SeparateResidualBlock(beta*(2**4), alpha=alpha)
        self.conv_block5 = ConvBN(beta*(2**5),kernel_size=(3,3), alpha=alpha,strides=2)
        self.block5_residual1 = SeparateResidualBlock(beta*(2**5), alpha=alpha)
        self.block5_residual2 = SeparateResidualBlock(beta*(2**5), alpha=alpha)
        self.block5_residual3 = SeparateResidualBlock(beta*(2**5), alpha=alpha)
        self.block5_residual4 = SeparateResidualBlock(beta*(2**5), alpha=alpha)
        self.output_layer = self.call(self.input_layer)

    def build(self, input_shape=None):
        if input_shape is None:
            input_shape = (None, self.h, self.w, self.c)
        super(NetBackbone, self).build(input_shape=input_shape)

    def call(self, inputs, training=None, mask=None):
        x = self.base_block(inputs,training=training)
        x = self.conv_block1(x,training=training)
        x = self.block1_residual1(x,training=training)
        x = self.conv_block2(x,training=training)
        x = self.block2_residual1(x,training=training)
        x = self.block2_residual2(x,training=training)
        x = self.conv_block3(x,training=training)
        x = self.block3_residual1(x,training=training)
        x = self.block3_residual2(x,training=training)
        x = self.block3_residual3(x,training=training)
        output_1 = self.block3_residual4(x,training=training)
        x = self.conv_block4(output_1,training=training)
        x = self.block4_residual1(x,training=training)
        x = self.block4_residual2(x,training=training)
        x = self.block4_residual3(x,training=training)
        output_2 = self.block4_residual4(x,training=training)
        x = self.conv_block5(output_2,training=training)
        x = self.block5_residual1(x,training=training)
        output_3 = self.block5_residual2(x,training=training)

        return [output_3, output_2, output_1]


class Net(tf.keras.models.Model, ABC):
    def __init__(self,input_shape,out_channels,alpha=0.1,beta=16):
        super(Net, self).__init__()
        self.h,self.w,self.channel = input_shape
        self.alpha = alpha
        self.beta = beta

        # Input Layer
        self.input_layer = tf.keras.layers.Input(shape=input_shape)

        # Backbone
        self.base_model = NetBackbone(input_shape,beta=self.beta)

        self.yolo_head_13 = YOLOHead(filters=self.beta*(2**5), out_channels=out_channels, alpha=self.alpha)
        self.up_sampling_13 = UpSampleLayer(filters=self.beta*(2**4), alpha=self.alpha)

        self.yolo_head_26 = YOLOHead(filters=self.beta*(2**4), out_channels=out_channels, alpha=self.alpha)
        self.up_sampling_26 = UpSampleLayer(filters=self.beta*(2**3), alpha=self.alpha)

        self.yolo_head_52 = YOLOHead(filters=self.beta*(2**3), out_channels=out_channels, alpha=self.alpha)

        # Output Layer
        self.output_layer = self.call(self.input_layer)

    def build(self, input_shape=None):
        if input_shape is None:
            input_shape = (None,self.w,self.h,self.channel)
        super(Net, self).build(input_shape=input_shape)

    def call(self,inputs,training=None, mask=None):
        x_13, x_26, x_52 = self.base_model(inputs, training=training)
        brand_13,output_end13 = self.yolo_head_13(x_13,training=training)
        brand_13 = self.up_sampling_13(brand_13,training=training)
        x_26 = tf.keras.layers.concatenate([brand_13,x_26])

        brand_26,output_end26 = self.yolo_head_26(x_26,training=training)
        brand_26 = self.up_sampling_26(brand_26,training=training)
        x_52 = tf.keras.layers.concatenate([brand_26,x_52])

        _,output_end52 = self.yolo_head_52(x_52,training=training)

        # print(output_end13.shape,output_end26.shape,output_end52.shape)
        # return size 13 26 52
        return [output_end13,output_end26,output_end52]


if __name__ == "__main__":
    """
    原始YOLOv3一共61646349个参数, DarkNet 一共40620640个参数.
    """
    # net = NetBackbone(input_shape=(416,416,3))
    net = Net(input_shape=(416, 416, 3), out_channels=3 * (5 + 14),beta=4)
    net.build()
    net.summary()



import torch
import torch.nn as nn
from inference_model.TransBraTS_old.Transformer import TransformerModel
from inference_model.TransBraTS_old.PositionalEncoding import FixedPositionalEncoding, LearnedPositionalEncoding
from inference_model.TransBraTS_old.Unet_skipconnection import Unet


class IDH_network(nn.Module):  # TODO: only this module is totally the same as the network structure depicted in the paper
    def __init__(self):
        # basic structure defined
        super(IDH_network, self).__init__()
        self.conv_x4_1_idh = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=1)
        self.conv_x4_1_grade = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=1)
        self.conv_en_idh = nn.Conv3d(in_channels=512, out_channels=256, kernel_size=1)
        self.conv_en_grade = nn.Conv3d(in_channels=512, out_channels=256, kernel_size=1)

        self.avg_pool_3d = nn.AvgPool3d(16, 1)
        self.max_pool_3d = nn.MaxPool3d(16, 1)
        self.drop_layer = nn.Dropout(p=0.2)

        self.Hidden_layer_1 = nn.Linear(640, 512)
        self.Hidden_layer_2 = nn.Linear(512, 32)
        self.classifier = nn.Linear(32, 2)
       
        self.Hidden_grade_1 = nn.Linear(640, 512)
        self.Hidden_grade_2 = nn.Linear(512, 32)
        self.grade_classifier = nn.Linear(32, 2)
 
        #self.softmax = nn.Softmax(dim=1)
    # inference path
    def forward(self, x4_1, encoder_output, idh, grade):
        # feature fusion (as depicted in Fig. 1 `High-level feature fusion`)
        x4_1_idh = self.conv_x4_1_idh(x4_1)
        x4_1_grade = self.conv_x4_1_grade(x4_1)

        encoder_idh = self.conv_en_idh(encoder_output)
        encoder_grade = self.conv_en_grade(encoder_output)

        x_idh = self.feature_fusion_layer(x4_1_idh,encoder_idh)
        x_grade = self.feature_fusion_layer(x4_1_grade,encoder_grade)

        # Dropout followed
        x_idh = self.drop_layer(x_idh)
        x_grade = self.drop_layer(x_grade)

        x_IDH_1 = self.Hidden_layer_1(x_idh)
        x_grade_1 = self.Hidden_grade_1(x_grade)

        x_IDH_2 = self.Hidden_layer_2(x_IDH_1)
        x_grade_2 = self.Hidden_grade_2(x_grade_1)

        y = self.classifier(x_IDH_2)
        y_grade = self.grade_classifier(x_grade_2)

        # return 0 or 1
        return y, y_grade

    def feature_fusion_layer(self,x4,encoder_output):
        # x4_1: OriginalImg_output of Unet downsample
        x4_avg = self.avg_pool_3d(x4)
        x4_max = self.max_pool_3d(x4)
        # encoder_avg: OriginalImg_output of Transformer-refined feature
        encoder_avg = self.avg_pool_3d(encoder_output)
        encoder_avg_max = self.max_pool_3d(encoder_output)
        # resize
        x4_avg = x4_avg.view(x4_avg.size(0), -1)
        x4_max = x4_max.view(x4_max.size(0), -1)
        # resize
        encoder_avg = encoder_avg.view(encoder_avg.size(0), -1)
        encoder_avg_max = encoder_avg_max.view(encoder_avg_max.size(0), -1)
        # concat
        return torch.cat([x4_avg,x4_max,encoder_avg,encoder_avg_max], dim=1)

# x1_1 shape: (N,16,128,128,128)
# x2_1 shape: (N,32,64,64,64)
# x3_1 shape: (N,64,32,32,32)
# x4_1 shape: (N, 128, 16, 16, 16)
# encoder_output  shape: (N,4096,512)
# intmd_encoder_outputs (N,4096,512)*7
# y4_1 shape (N, 128, 16, 16, 16)

class Decoder_modual(nn.Module):
    def __init__(self):
        super(Decoder_modual, self).__init__()

        self.embedding_dim = 512

        self.Enblock8_1 = EnBlock1(in_channels=self.embedding_dim)

        self.Enblock8_2 = EnBlock2(in_channels=self.embedding_dim // 4)

        self.DeUp4 = DeUp_Cat(in_channels=self.embedding_dim // 4, out_channels=self.embedding_dim // 8)
        self.DeBlock4 = DeBlock(in_channels=self.embedding_dim // 8)

        self.DeUp3 = DeUp_Cat(in_channels=self.embedding_dim // 8, out_channels=self.embedding_dim // 16)
        self.DeBlock3 = DeBlock(in_channels=self.embedding_dim // 16) #32

        self.DeUp2 = DeUp_Cat(in_channels=self.embedding_dim // 16, out_channels=self.embedding_dim // 32)
        self.DeBlock2 = DeBlock(in_channels=self.embedding_dim // 32)  #16

        self.endconv = nn.Conv3d(self.embedding_dim // 32, 4, kernel_size=1)

        self.Softmax = nn.Softmax(dim=1)

    def forward(self,x1_1, x2_1, x3_1, x8):
        return self.decode(x1_1, x2_1, x3_1, x8)

    def decode(self, x1_1, x2_1, x3_1, x8):

        x8 = self.Enblock8_1(x8)
        y4_1 = self.Enblock8_2(x8)

        y4 = self.DeUp4(y4_1, x3_1)  # (1, 64, 32, 32, 32)
        y3_1 = self.DeBlock4(y4)

        y3 = self.DeUp3(y3_1, x2_1)  # (1, 32, 64, 64, 64)
        y2_1 = self.DeBlock3(y3)

        y2 = self.DeUp2(y2_1, x1_1)  # (1, 16, 128, 128, 128)
        y2 = self.DeBlock2(y2)

        y = self.endconv(y2)      # (1, 4, 128, 128, 128)

        y = self.Softmax(y)

        return y


class TransformerBraTS(nn.Module):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        conv_patch_representation=True,
        positional_encoding_type="learned",
    ):
        super(TransformerBraTS, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0

        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.conv_patch_representation = conv_patch_representation

        self.num_patches = int((img_dim // patch_dim) ** 3)
        self.seq_length = self.num_patches
        self.flatten_dim = 128 * num_channels

        self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)  # positional encoding dropout

        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,
            # dropout in Transformer
            self.dropout_rate,
            self.attn_dropout_rate,
        )

        # LayerNorm
        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        if self.conv_patch_representation:
            self.conv_x = nn.Conv3d(
                128,
                self.embedding_dim,
                kernel_size=3,
                stride=1,
                padding=1
            )

        self.Unet = Unet(in_channels=4, base_channels=16, num_classes=4)
        self.bn = nn.BatchNorm3d(128)
        self.relu = nn.LeakyReLU(inplace=True)

    def encode(self, x):
        if self.conv_patch_representation:
            # combine embedding with conv patch distribution
            x1_1, x2_1, x3_1, x4_1 = self.Unet(x)
            x = self.bn(x4_1)
            x = self.relu(x)
            x = self.conv_x(x)

            x = x.permute(0, 2, 3, 4, 1).contiguous()
            x = x.view(x.size(0), -1, self.embedding_dim)
            # (1, 128, 16, 16, 16)
            # x.shape (N, B, C)
        else:
            # TODO: this branch seemed to be meaningless
            x = self.Unet(x)
            x = self.bn(x)
            x = self.relu(x)
            x = (
                x.unfold(2, 2, 2)
                .unfold(3, 2, 2)
                .unfold(4, 2, 2)
                .contiguous()
            )
            x = x.view(x.size(0), x.size(1), -1, 8)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size(0), -1, self.flatten_dim)
            x = self.linear_encoding(x)

        x = self.position_encoding(x)
        x = self.pe_dropout(x)

        # apply transformer
        x, intmd_x = self.transformer(x)

        x = self.pre_head_ln(x)

        intmd_layers = [1, 2, 3, 4]
        # assert intmd_layers is not None, "pass the intermediate layers for MLA"
        # encoder_outputs = {}
        #all_keys = []
        # for i in intmd_layers:
        #     val = str(2 * i - 1)
        #     _key = 'Z' + str(i)
        #     all_keys.append(_key)
        #     encoder_outputs[_key] = intmd_x[val]
        # all_keys.reverse()

        #x8 = encoder_outputs[all_keys[0]]
        # x8 = intmd_x['7']
        x8 = self._reshape_output(x)
        # print("x8 shape:", x8.shape)

        return x1_1, x2_1, x3_1,x4_1, x8

    def decode(self, x):
        # TODO: wrong message, not implemented in child class !!!
        raise NotImplementedError("Should be implemented in child class!!")

    def forward(self, x):

        x1_1, x2_1, x3_1,x4_1, encoder_output = self.encode(x)
        return x1_1, x2_1, x3_1, x4_1, encoder_output
        # return x1_1, x2_1, x3_1,x4_1, encoder_output
        # x1_1 shape: (N,16,128,128,128)
        # x2_1 shape: (N,32,64,64,64)
        # x3_1 shape: (N,64,32,32,32)
        # x4_1 shape: (N, 128, 16, 16, 16)
        # encoder_output  shape: (N,4096,512)
        # intmd_encoder_outputs (N,4096,512)*7
        # y4_1 shape (N, 128, 16, 16, 16)

        # print("intmd_encoder_outputs:",intmd_encoder_outputs)
        # print("intmd_encoder_outputs['0']:", intmd_encoder_outputs['0'])
        # y4_1,decoder_output = self.decode(
        #    x1_1, x2_1, x3_1, encoder_output, intmd_encoder_outputs, auxillary_output_layers)
        # IDH_out = self.IDH_classifier(x4_1, encoder_output,y4_1)
        # print("encoder_output:",encoder_output.shape)
        # if auxillary_output_layers is not None:
        #    auxillary_outputs = {}
        #    for i in auxillary_output_layers:
        #        val = str(2 * i - 1)
        #        _key = 'Z' + str(i)
        #        auxillary_outputs[_key] = intmd_encoder_outputs[val]
        #
        #    return IDH_out,decoder_output
        #
        # return IDH_out,decoder_output
    def get_last_shared_layer(self):
        return self.pre_head_ln

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x

    # def IDH_classifier(self,x4_1, encoder_output,y4_1 ):
    #     """
    #
    #     :param x4_1: (N, 128, 16, 16, 16)
    #     :param encoder_output: (N,4096,512)
    #     :param y4_1:(N, 128, 16, 16, 16)
    #     :return:
    #     """
    #
    #     x4_1 = self.avg_pool_3d_1(x4_1)
    #     x4_1 = x4_1.view(x4_1.size(0),-1)
    #     y4_1 = self.avg_pool_3d_2(y4_1)
    #     y4_1 = y4_1.view(y4_1.size(0), -1)
    #     encoder_output = encoder_output.mean(axis=1)
    #     feature_fusion = torch.cat([x4_1,encoder_output,y4_1],dim=1)
    #     x = self.Hidden_layer(feature_fusion)
    #     x = self.bn_layer(x)
    #     x= self.classifier(x)
    #     y = self.sigmoid(x)
    #     #print ("y shape:", y.shape)
    #     return y

class BraTS(TransformerBraTS):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        conv_patch_representation=True,
        positional_encoding_type="learned",
    ):
        super(BraTS, self).__init__(
            img_dim=img_dim,
            patch_dim=patch_dim,
            num_channels=num_channels,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            conv_patch_representation=conv_patch_representation,
            positional_encoding_type=positional_encoding_type,
        )

        self.num_classes = num_classes

        # self.Softmax = nn.Softmax(dim=1)
        #
        # self.Enblock8_1 = EnBlock1(in_channels=self.embedding_dim)
        # self.Enblock8_2 = EnBlock2(in_channels=self.embedding_dim // 4)
        #
        # self.DeUp4 = DeUp_Cat(in_channels=self.embedding_dim//4, out_channels=self.embedding_dim//8)
        # self.DeBlock4 = DeBlock(in_channels=self.embedding_dim//8)
        #
        # self.DeUp3 = DeUp_Cat(in_channels=self.embedding_dim//8, out_channels=self.embedding_dim//16)
        # self.DeBlock3 = DeBlock(in_channels=self.embedding_dim//16)
        #
        # self.DeUp2 = DeUp_Cat(in_channels=self.embedding_dim//16, out_channels=self.embedding_dim//32)
        # self.DeBlock2 = DeBlock(in_channels=self.embedding_dim//32)
        # self.endconv = nn.Conv3d(self.embedding_dim // 32, 4, kernel_size=1)

class EnBlock1(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock1, self).__init__()
        self.bn1 = nn.BatchNorm3d(512)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(512 // 4)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels // 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = self.conv2(x1)
        return x1


class EnBlock2(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock2, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(512 // 4)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(512 // 4)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = self.conv2(x1)
        x1 = x1 + x
        return x1

class DeUp_Cat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeUp_Cat, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(out_channels*2, out_channels, kernel_size=1)

    def forward(self, x, prev):
        x1 = self.conv1(x)
        y = self.conv2(x1)
        # y = y + prev
        y = torch.cat((prev, y), dim=1)
        y = self.conv3(y)
        return y

class DeBlock(nn.Module):
    def __init__(self, in_channels):
        super(DeBlock, self).__init__()

        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(in_channels)
        self.relu2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = self.conv2(x1)
        x1 = x1 + x
        return x1




def TransBraTS(dataset='brats', _conv_repr=True, _pe_type="learned"):

    if dataset.lower() == 'brats':
        img_dim = 128
        num_classes = 4

    num_channels = 4
    patch_dim = 8
    aux_layers = [1, 2, 3, 4]
    model = BraTS(
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim=512,
        num_heads=8,
        num_layers=4, #4
        hidden_dim=4096,   #原4096
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        conv_patch_representation=_conv_repr,
        positional_encoding_type=_pe_type,
    )
    return model


if __name__ == '__main__':
    with torch.no_grad():
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((1, 4, 128, 128, 128), device=cuda0)
        model = TransBraTS(dataset='brats', _conv_repr=True, _pe_type="learned")
        model.cuda()
        y = model(x)
        print(y.shape)

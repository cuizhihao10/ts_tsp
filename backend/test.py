import nibabel as nib
import os
import numpy as np
import imageio

def test():
    seg_img = nib.load('./upload/BraTS20_Training_232_seg.nii')
    seg_img_data = seg_img.get_fdata()
    seg_img_data = seg_img_data.transpose(1, 0, 2)
    Snapshot_img = np.zeros(shape=(240, 240, 3, 155), dtype=np.uint8)
    Snapshot_img[:, :, 0, :][np.where(seg_img_data == 1)] = 255
    Snapshot_img[:, :, 1, :][np.where(seg_img_data == 2)] = 255
    Snapshot_img[:, :, 2, :][np.where(seg_img_data == 3)] = 255
    Snapshot_img[:, :, 2, :][np.where(seg_img_data == 4)] = 255

    for frame in range(155):  # 每一个切片保存一张彩色图像
        if not os.path.exists(os.path.join('./inference_model/output/visualization/', '111')):
            os.makedirs(os.path.join('./inference_model/output/visualization/', '111'))
        # scipy.misc.imsave(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])
        imageio.imwrite(os.path.join('./inference_model/output/visualization/', '111', str(frame) + '.png'), Snapshot_img[:, :, :, frame])
# print(seg_img_data.shape)
# print(seg_img_data[0][0][0])
# print(seg_img_data[0][0][1])
# print(seg_img_data[0][0][2])
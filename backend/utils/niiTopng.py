import SimpleITK as sitk
import numpy as np
from PIL import Image
import imageio
import os


def nii2png(img_path):
    filename = img_path.split('/')[-2].split('\\')[-1]
    files = os.listdir(img_path)
    for file in files:
        I = sitk.ReadImage(os.path.join(img_path, file))
        rescakFilter = sitk.RescaleIntensityImageFilter()
        rescakFilter.SetOutputMaximum(255)
        rescakFilter.SetOutputMinimum(0)
        I = rescakFilter.Execute(I)
        img = sitk.GetArrayFromImage(I)
        for i in range(155):
            img_slice = img[i, :, :]
            # img_slice = img_slice * 255
            img_slice = img_slice.astype(np.uint8)
            img_slice = Image.fromarray(img_slice)
            if not os.path.exists(os.path.join('./OriginalImg_output/', filename, file.split('.')[0])):
                os.makedirs(os.path.join('./OriginalImg_output/', filename, file.split('.')[0]))
            imageio.imwrite(os.path.join('./OriginalImg_output/', filename, file.split('.')[0], str(i)
                                    + '.png'), img_slice)
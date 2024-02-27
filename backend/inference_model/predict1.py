import os
import torch
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
import torch.optim
import torch.nn.functional as F
import numpy as np
import imageio
import SimpleITK as sitk



def caijian(a, b, c):
    (zstart, ystart, xstart), (zstop, ystop, xstop) = c.min(axis=-1), c.max(axis=-1) + 1
    roi_image = a[zstart:zstop, ystart:ystop, xstart:xstop]
    roi_mask = b[zstart:zstop, ystart:ystop, xstart:xstop]
    roi_image[roi_mask == 0] = 0
    # plt.imshow(roi_image[:,:,25],'gray')
    # plt.show()
    return roi_image


def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  ##获取原图size
    originSpacing = itkimage.GetSpacing()  ##获取原图spacing
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(int)  ##spacing格式转换

    resampler.SetReferenceImage(itkimage)  ##指定需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  ##得到重新采样后的图像

    return itkimgResampled


def find_center(image):
    # Create a copy of the image to modify
    new_img = np.copy(image)
    a, b, c = image.shape

    # Iterate through all points in the image
    for x in range(a):
        for y in range(b):
            for z in range(c):
                # Check if the current point is non-zero
                if image[x, y, z] != 0:
                    # Initialize the sum with the value of the current point
                    value_sum = image[x, y, z]
                    # Sum the values of adjacent non-zero pixels in all directions
                    for dx in [-1, 1]:
                        i = x
                        while 0 <= i + dx < a and image[i + dx, y, z] != 0:
                            i += dx
                            value_sum += image[i, y, z]
                    for dy in [-1, 1]:
                        j = y
                        while 0 <= j + dy < b and image[x, j + dy, z] != 0:
                            j += dy
                            value_sum += image[x, j, z]
                    for dz in [-1, 1]:
                        k = z
                        while 0 <= k + dz < c and image[x, y, k + dz] != 0:
                            k += dz
                            value_sum += image[x, y, k]

                    # Assign the computed sum to the current point in the new image
                    new_img[x, y, z] = value_sum

    return new_img


def post_crop(image, max_value):
    # Find the maximum value in the image
    a, b, c = image.shape
    print(f"The maximum pixel value is: {max_value}")

    # Get the indices of the maximum value
    max_indices = np.argwhere(image == max_value)
    # Create a mask with the same shape as the image initialized to zero
    mask = np.zeros_like(image)

    while True:
        # Update the adjacent non-zero pixels to the maximum value
        max_sum_old = np.count_nonzero(mask == max_value)
        for idx in max_indices:
            x, y, z = idx
            mask[x, y, z] = max_value
            # Spread the maximum value to adjacent non-zero pixels in all directions
            for dx in [-1, 1]:
                i = x
                while 0 <= i + dx < a and image[i + dx, y, z] != 0:
                    i += dx
                    mask[i, y, z] = max_value
            for dy in [-1, 1]:
                j = y
                while 0 <= j + dy < b and image[x, j + dy, z] != 0:
                    j += dy
                    mask[x, j, z] = max_value
            for dz in [-1, 1]:
                k = z
                while 0 <= k + dz < c and image[x, y, k + dz] != 0:
                    k += dz
                    mask[x, y, k] = max_value
        max_sum_new = np.count_nonzero(mask == max_value)
        max_indices = np.argwhere(mask == max_value)
        print(max_sum_new)
        if max_sum_new == max_sum_old:
            break
    return mask


def tailor_and_concat(x, model, idh, grade, mgmt, pq):
    temp = []
    idh_temp=[]
    grade_temp=[]
    mgmt_temp=[]
    pq_temp=[]
    temp.append(x[..., :128, :128, :128])
    temp.append(x[..., :128, 112:240, :128])
    temp.append(x[..., 112:240, :128, :128])
    temp.append(x[..., 112:240, 112:240, :128])
    temp.append(x[..., :128, :128, 27:155])
    temp.append(x[..., :128, 112:240, 27:155])
    temp.append(x[..., 112:240, :128, 27:155])
    temp.append(x[..., 112:240, 112:240, 27:155])

    y = x.clone()
    #y = torch.cat((x,x[:,[0],:,:,:]),dim=1).clone()
    for i in range(len(temp)):
        x1_1, x2_1, x3_1, x4_1, encoder_output = model['en'](temp[i])

        seg_output = model['seg'](x1_1, x2_1, x3_1, encoder_output)

        idh_out, grade_out, mgmt_out, pq_out = model['idh'](x4_1, encoder_output, idh, grade, mgmt, pq)

        temp[i] = seg_output

        idh_temp.append(idh_out)
        grade_temp.append(grade_out)
        mgmt_temp.append(mgmt_out)
        pq_temp.append(pq_out)

    y[..., :128, :128, :128] = temp[0]
    y[..., :128, 128:240, :128] = temp[1][..., :, 16:128, :]
    y[..., 128:240, :128, :128] = temp[2][..., 16:128, :, :]
    y[..., 128:240, 128:240, :128] = temp[3][..., 16:128, 16:128, :]
    y[..., :128, :128, 128:155] = temp[4][..., 96:123]
    y[..., :128, 128:240, 128:155] = temp[5][..., :, 16:128, 96:123]
    y[..., 128:240, :128, 128:155] = temp[6][..., 16:128, :, 96:123]
    y[..., 128:240, 128:240, 128:155] = temp[7][..., 16:128, 16:128, 96:123]

    idh_out = torch.mean(torch.stack(idh_temp), dim=0)
    grade_out = torch.mean(torch.stack(grade_temp), dim=0)
    return y[..., :155], idh_out, grade_out, mgmt_out, pq_out


def dice_coeff(pred, target, label):
    smooth = 1.
    pred_2 = pred.copy()
    target_2 = target.copy()
    pred_2[np.where(pred != label)] = 0
    pred_2[np.where(pred == label)] = 1
    target_2[np.where(target != label)] = 0
    target_2[np.where(target == label)] = 1
    y_true_f = target_2.flatten()
    y_pred_f = pred_2.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_coeff_all(pred, target):
    smooth = 1.
    pred_1 = pred.copy()
    target_1 = target.copy()
    pred_1[np.where(pred == 2)] = 1
    pred_1[np.where(pred == 4)] = 1
    target_1[np.where(target == 2)] = 1
    target_1[np.where(target == 4)] = 1
    y_true_f = target_1.flatten()
    y_pred_f = pred_1.flatten()
    print(y_true_f.shape, y_pred_f.shape)
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_coeff_TC(pred, target):
    smooth = 1.
    pred_3 = pred.copy()
    target_3 = target.copy()
    pred_3[np.where(pred == 4)] = 1
    target_3[np.where(target == 4)] = 1
    pred_3[np.where(pred == 2)] = 0
    target_3[np.where(target == 2)] = 0
    y_true_f = target_3.flatten()
    y_pred_f = pred_3.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def validate_softmax(
        valid_loader,
        model,
        # load_file,
        # multimodel,
        savepath='',  # when in validation set, you must specify the path to save the 'nii' segmentation results here
        names=None,  # The names of the patients orderly。   shuffle = false所以直接用i提取dataset中的name即可！
        use_TTA=False,  # Test time augmentation, False as default!
        save_format=None,  # ['nii','npy'], use 'nii' as default. Its purpose is for submission.
        snapshot=False,  # for visualization. Default false. It is recommended to generate the visualized figures.
        visual='',  # the path to save visualization
        postprocess=False,  # Default False, when use postprocess, the score of dice_ET would be changed.
        valid_in_train=False,  # if you are valid when train
):
    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
    import pandas as pd
    H, W, T = 240, 240, 155

    model['en'].eval()
    model['seg'].eval()
    model['idh'].eval()
    # model['grade'].eval()

    runtimes = []
    ET_voxels_pred_list = []

    grade_prob1 = []
    grade_conf = []
    grade_class = []
    grade_truth = []
    grade_error_case = []

    idh_prob1 = []
    idh_conf = []
    idh_class = []
    idh_truth = []
    idh_error_case = []

    mgmt_prob1 = []
    mgmt_conf = []
    mgmt_class = []
    mgmt_truth = []
    mgmt_error_case = []

    pq_prob1 = []
    pq_conf = []
    pq_class = []
    pq_truth = []
    pq_error_case = []

    ids = []

    NCR_ratio = []
    ED_ratio = []
    ET_ratio = []
    dice_scores = []
    dice_scores_label_2 = []
    dice_scores_label_1_4 = []
    postprocess = False

    for i, data in enumerate(valid_loader):
        print('-------------------------------------------------------------------')
        msg = 'Subject {}/{}, '.format(i + 1, len(valid_loader))

        # t1ce_Path = '/public/home/hpc226511030/Archive/BraTS2021_TrainingData/MICCAI_BraTS2021_TrainingData/BraTS2021_01061/BraTS2021_01061_t1ce.nii.gz'
        # t1ce_image = nib.load(t1ce_Path)
        # print(len(data))
        # print("data[0]:",data[0].shape,'data[1]',data[1],'data[2]',data[2] )
        data = [t.cuda(non_blocking=True) for t in data]
        x = data[0]
        idh = data[1]
        grade = data[2]
        mgmt = data[3]
        pq = data[4]

        # else:
        #     x = data
        #     x.cuda()

        x = x[..., :155]

        TTA_1, TTA_2, TTA_3, TTA_4, TTA_5, TTA_6, TTA_7, TTA_8 = tailor_and_concat(x, model, idh, grade, mgmt,
                                                                                   pq), tailor_and_concat(
            x.flip(dims=(2,)), model, idh, grade, mgmt, pq), \
            tailor_and_concat(x.flip(dims=(3,)), model, idh, grade, mgmt, pq), tailor_and_concat(x.flip(dims=(4,)),
                                                                                                 model, idh, grade,
                                                                                                 mgmt, pq), \
            tailor_and_concat(x.flip(dims=(2, 3)), model, idh, grade, mgmt, pq), tailor_and_concat(x.flip(dims=(2, 4)),
                                                                                                   model, idh, grade,
                                                                                                   mgmt, pq), \
            tailor_and_concat(x.flip(dims=(3, 4)), model, idh, grade, mgmt, pq), tailor_and_concat(
            x.flip(dims=(2, 3, 4)), model, idh, grade, mgmt, pq)

        # todo 以下为分割的TTA,翻转了预测的结果,再翻转回去求平均
        logit = F.softmax(TTA_1[0], 1)  # no flip
        logit += F.softmax(TTA_2[0].flip(dims=(2,)), 1)  # flip H
        logit += F.softmax(TTA_3[0].flip(dims=(3,)), 1)  # flip W
        logit += F.softmax(TTA_4[0].flip(dims=(4,)), 1)  # flip D
        logit += F.softmax(TTA_5[0].flip(dims=(2, 3)), 1)  # flip H, W
        logit += F.softmax(TTA_6[0].flip(dims=(2, 4)), 1)  # flip H, D
        logit += F.softmax(TTA_7[0].flip(dims=(3, 4)), 1)  # flip W, D
        logit += F.softmax(TTA_8[0].flip(dims=(2, 3, 4)), 1)  # flip H, W, D
        output = logit / 8.0  # TTA
        idh_probs = []
        grade_probs = []
        mgmt_probs = []
        pq_probs = []
        for pred in [TTA_1, TTA_2, TTA_3, TTA_4, TTA_5, TTA_6, TTA_7, TTA_8]:
            idh_probs.append(F.softmax(pred[1], 1))
            grade_probs.append(F.softmax(pred[2], 1))
            mgmt_probs.append(F.softmax(pred[3], 1))
            pq_probs.append(F.softmax(pred[4], 1))

        idh_pred = torch.mean(torch.stack(idh_probs), dim=0)
        print("idh_pred:", idh_pred)
        grade_pred = torch.mean(torch.stack(grade_probs), dim=0)
        print("grade_pred:", grade_pred)
        mgmt_pred = torch.mean(torch.stack(mgmt_probs), dim=0)
        print("mgmt_pred:", mgmt_pred)
        pq_pred = torch.mean(torch.stack(pq_probs), dim=0)
        print("pq_pred:", pq_pred)

        idh_pred_class = torch.argmax(idh_pred, dim=1)  # 类别概率值还需要比一个大小。
        idh_class.append(idh_pred_class.item())
        idh_prob1.append(idh_pred[0][1].item())
        idh_conf.append(idh_pred[0][idh_pred_class.item()])  # 如果是tensor，则[0]先是一整个tensor

        print('id:', names[i], 'IDH_truth:', idh.item(), 'IDH_pred:', idh_pred_class.item())
        #
        grade_truth.append(grade.item())
        grade_pred_class = torch.argmax(grade_pred, dim=1)
        grade_class.append(grade_pred_class.item())
        grade_prob1.append(grade_pred[0][1].item())
        grade_conf.append(grade_pred[0][grade_pred_class.item()])
        print('id:', names[i], 'grade_truth:', grade.item(), 'grade_pred:', grade_pred_class.item())
        #
        mgmt_truth.append(mgmt.item())
        mgmt_pred_class = torch.argmax(mgmt_pred, dim=1)
        mgmt_class.append(mgmt_pred_class.item())
        mgmt_prob1.append(mgmt_pred[0][1].item())
        mgmt_conf.append(mgmt_pred[0][mgmt_pred_class.item()])
        print('id:', names[i], 'mgmt_truth:', mgmt.item(), 'mgmt_pred:', mgmt_pred_class.item())
        #
        pq_truth.append(pq.item())
        pq_pred_class = torch.argmax(pq_pred, dim=1)
        pq_class.append(pq_pred_class.item())
        pq_prob1.append(pq_pred[0][1].item())
        pq_conf.append(pq_pred[0][pq_pred_class.item()])
        print('id:', names[i], 'pq_truth:', pq.item(), 'pq_pred:', pq_pred_class.item())

        ids.append(names[i])
        idh_truth.append(idh.item())
        if not (idh_pred_class.item() == idh.item()):
            idh_error_case.append({'id': names[i], 'truth:': idh.item(), 'pred': idh_pred_class.item()})
        if not (grade_pred_class.item() == grade.item()):
            grade_error_case.append({'id': names[i], 'truth:': grade.item(), 'pred': grade_pred_class.item()})
        if not (mgmt_pred_class.item() == mgmt.item()):
            mgmt_error_case.append({'id': names[i], 'truth:': mgmt.item(), 'pred': mgmt_pred_class.item()})
        if not (pq_pred_class.item() == pq.item()):
            pq_error_case.append({'id': names[i], 'truth:': pq.item(), 'pred': pq_pred_class.item()})

        output = output[0, :, :H, :W, :T].cpu().detach().numpy()
        output = output.argmax(0)  # 分割结果的四张特征图的最大概率值所在标签（第二个维度-0/1/2/3）
        # seg_img[np.where(output == 1)] = 1
        # seg_img[np.where(output == 2)] = 2
        # seg_img[np.where(output == 3)] = 4

        name = str(i)
        if names:
            name = names[i]
            msg += '{:>20}, '.format(name)

        print(msg)

        if savepath:
            # .npy for further model ensemble
            # .nii for directly model submission
            assert save_format in ['npy', 'nii']
            if save_format == 'npy':
                np.save(os.path.join(savepath, name + '_preds'), output)
            if save_format == 'nii':
                # raise NotImplementedError
                oname = os.path.join(savepath, name, name + '_pseudo_seg.nii.gz')
                seg_img = np.zeros(shape=(H, W, T), dtype=np.uint8)

                seg_img[np.where(output == 1)] = 1
                seg_img[np.where(output == 2)] = 2
                seg_img[np.where(output == 3)] = 4

                if postprocess:
                    image = seg_img.copy()
                    image[np.where(seg_img == 2)] = 1
                    image[np.where(seg_img == 4)] = 1

                    zuobiao = np.where(image != 0)
                    zuobiao = np.array(zuobiao)
                    WT = caijian(image, image, zuobiao)
                    itkimage = sitk.GetImageFromArray(WT)
                    itkimgResampled = resize_image_itk(itkimage, (128, 128, 128),
                                                       resamplemethod=sitk.sitkLinear)  ## resample使用线性插值
                    WT_img = sitk.GetArrayFromImage(itkimgResampled)
                    # Calculate the pixel density
                    A, B, C = WT_img.shape
                    pixel_density = np.sum(WT_img) / (A * B * C)
                    print(f"The pixel density is: {pixel_density}")

                    max_value = []
                    mask = []
                    # Execute the functions as per the updated requirement
                    if pixel_density < 0.5:
                        print('I am entered.')
                        centered_img = find_center(image)
                        max_value.append(np.max(centered_img))
                        mask.append(post_crop(centered_img, max_value[-1]))
                        # crop
                        seg_img_1 = seg_img.copy()
                        pre_sum = np.count_nonzero(seg_img_1)
                        print('pre_sum:', pre_sum)
                        seg_img_1[np.where(mask[-1] != max_value[-1])] = 0
                        post_sum = np.count_nonzero(seg_img_1)
                        print('post_sum——1:', post_sum)
                        if post_sum / pre_sum < 0.6:
                            while post_sum / pre_sum < 0.6:
                                print('do it again.')
                                centered_img[np.where(mask[-1] == max_value[-1])] = 0
                                print('next value:', np.max(centered_img))
                                max_value.append(np.max(centered_img))
                                mask.append(post_crop(centered_img, max_value[-1]))
                                print(np.count_nonzero(mask[0]), np.count_nonzero(mask[-1]))

                                seg_img_2 = seg_img.copy()
                                seg_img_3 = seg_img.copy()

                                for i, _ in enumerate(zip(mask, max_value)):
                                    seg_img_2[np.where(mask[i] == max_value[i])] = 100

                                seg_img_3[np.where(seg_img_2 != 100)] = 0
                                post_sum = np.count_nonzero(seg_img_3)
                                print('post_sum——2:', post_sum)
                                # break
                            else:
                                seg_img = seg_img_3
                        else:
                            seg_img = seg_img_1

                print('NCR:', np.sum(seg_img == 1), ' | ED:', np.sum(seg_img == 2), ' | ET:', np.sum(seg_img == 4))
                WT_vol = np.sum((seg_img == 1) | (seg_img == 2) | (seg_img == 4))
                # print('WT:', WT_vol , ' | TC:', np.sum((seg_img == 1) | (seg_img == 4)))

                NCR_ratio.append(round(np.sum(seg_img == 1) / WT_vol, 2))
                print('NCR_ratio:', NCR_ratio)
                ED_ratio.append(round(np.sum(seg_img == 2) / WT_vol, 2))
                ET_ratio.append(round(np.sum(seg_img == 4) / WT_vol, 2))
                #
                # if torch.max(target) > 0.5:
                #     # todo 计算dice
                #     target = target.cpu().detach().numpy()
                #     dice_score = dice_coeff_all(seg_img, target)
                #     dice_scores.append(dice_score.item())  # 将 DICE 值添加到列表中
                #     print('DICE:', dice_score.item())
                #
                #     # 计算标签为2的DICE值
                #     dice_score_label_2 = dice_coeff(seg_img, target, label=2)
                #     dice_scores_label_2.append(dice_score_label_2.item())  # 将标签为2的DICE值添加到列表中
                #     print('DICE_label_2:', dice_score_label_2.item())
                #
                #     # 计算标签为1和4的DICE值
                #     dice_score_label_1_4 = dice_coeff_TC(seg_img, target)
                #     dice_scores_label_1_4.append(dice_score_label_1_4.item())  # 将标签为1和4的DICE值添加到列表中
                #     print('DICE_label_1_4:', dice_score_label_1_4.item())
                #
                # # nib.save(nib.Nifti1Image(seg_img, affine=t1ce_image.affine, header=t1ce_image.header), oname)
                # print('Successfully save {}'.format(oname))

                if snapshot:
                    """ --- grey figure---"""
                    # Snapshot_img = np.zeros(shape=(H,W,T),dtype=np.uint8)
                    # Snapshot_img[np.where(output[1,:,:,:]==1)] = 64
                    # Snapshot_img[np.where(output[2,:,:,:]==1)] = 160
                    # Snapshot_img[np.where(output[3,:,:,:]==1)] = 255
                    """ --- colorful figure--- """
                    seg_img = seg_img.transpose(1, 0, 2)
                    Snapshot_img = np.zeros(shape=(H, W, 3, T), dtype=np.uint8)
                    Snapshot_img[:, :, 0, :][np.where(seg_img == 1)] = 255
                    Snapshot_img[:, :, 1, :][np.where(seg_img == 2)] = 255
                    Snapshot_img[:, :, 2, :][np.where(seg_img == 4)] = 255

                    for frame in range(T):  # 每一个切片保存一张彩色图像
                        if not os.path.exists(os.path.join(visual, name)):
                            os.makedirs(os.path.join(visual, name))
                        # scipy.misc.imsave(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])
                        imageio.imwrite(os.path.join(visual, name, str(frame) + '.png'), Snapshot_img[:, :, :, frame])

    return {"idh_pred": idh_pred,
            "grade_pred": grade_pred,
            "pq_pred": pq_pred,
            "idh_class": idh_pred_class,
            "grade_class": grade_pred_class,
            "pq_class": pq_pred_class,
            "NCR_ratio": NCR_ratio[0],
            "ED_ratio": ED_ratio[0],
            "ET_ratio": ET_ratio[0]}
















# def tailor_and_concat(x, model, idh, grade):
#     temp = []
#     idh_temp=[]
#     grade_temp=[]
#     temp.append(x[..., :128, :128, :128])
#     temp.append(x[..., :128, 112:240, :128])
#     temp.append(x[..., 112:240, :128, :128])
#     temp.append(x[..., 112:240, 112:240, :128])
#     temp.append(x[..., :128, :128, 27:155])
#     temp.append(x[..., :128, 112:240, 27:155])
#     temp.append(x[..., 112:240, :128, 27:155])
#     temp.append(x[..., 112:240, 112:240, 27:155])
#
#     y = x.clone()
#     #y = torch.cat((x,x[:,[0],:,:,:]),dim=1).clone()
#     for i in range(len(temp)):
#
#         x1_1, x2_1, x3_1,x4_1, encoder_output = model['en'](temp[i])
#
#         seg_output = model['seg'](x1_1, x2_1, x3_1,encoder_output)
#
#         idh_out, grade_out = model['idh'](x4_1, encoder_output, idh, grade)
#
#         temp[i] = seg_output
#         idh_temp.append(idh_out)
#         grade_temp.append(grade_out)
#
#     y[..., :128, :128, :128] = temp[0]
#     y[..., :128, 128:240, :128] = temp[1][..., :, 16:128, :]
#     y[..., 128:240, :128, :128] = temp[2][..., 16:128, :, :]
#     y[..., 128:240, 128:240, :128] = temp[3][..., 16:128, 16:128, :]
#     y[..., :128, :128, 128:155] = temp[4][..., 96:123]
#     y[..., :128, 128:240, 128:155] = temp[5][..., :, 16:128, 96:123]
#     y[..., 128:240, :128, 128:155] = temp[6][..., 16:128, :, 96:123]
#     y[..., 128:240, 128:240, 128:155] = temp[7][..., 16:128, 16:128, 96:123]
#
#     idh_out = torch.mean(torch.stack(idh_temp), dim=0)
#     grade_out = torch.mean(torch.stack(grade_temp), dim=0)
#     return y[..., :155], idh_out, grade_out
#
# def validate_softmax(
#         valid_loader,
#         model,
#         # load_file,
#         # multimodel,
#         savepath='',  # when in validation set, you must specify the path to save the 'nii' segmentation results here
#         names=None,  # The names of the patients orderly。   shuffle = false所以直接用i提取dataset中的name即可！
#         # use_TTA=False,  # Test time augmentation, False as default!
#         save_format=None,  # ['nii','npy'], use 'nii' as default. Its purpose is for submission.
#         snapshot=True,  # for visualization. Default false. It is recommended to generate the visualized figures.
#         visual='',  # the path to save visualization
#         # postprocess=False,  # Default False, when use postprocess, the score of dice_ET would be changed.
#         # valid_in_train=False,  # if you are valid when train
#         ):
#
#     H, W, T = 240, 240, 155
#
#     model['en'].eval()
#     model['seg'].eval()
#     model['idh'].eval()
#
#     runtimes = []
#     ET_voxels_pred_list = []
#
#     grade_prob1 = []
#     grade_conf = []
#     grade_class = []
#     grade_truth = []
#     grade_error_case = []
#
#     idh_prob1 = []
#     idh_conf = []
#     idh_class = []
#     idh_truth = []
#     idh_error_case = []
#     ids = []
#
#     NCR_ratio =[]
#     ED_ratio =[]
#     ET_ratio = []
#
#     for i, data in enumerate(valid_loader):
#         print('-------------------------------------------------------------------')
#         msg = 'Subject {}/{}, '.format(i + 1, len(valid_loader))
#
#         print(len(data))
#         print("data[0]:",data[0].shape,'data[1]',data[1],'data[2]',data[2] )
#         data = [t.cuda(non_blocking=True) for t in data]
#         x = data[0]
#         grade = data[1]
#         idh = data[2]
#
#         # else:
#         #     x = data
#         #     x.cuda()
#
#         x = x[..., :155]
#
#         TTA_1,TTA_2,TTA_3,TTA_4,TTA_5,TTA_6,TTA_7,TTA_8 = tailor_and_concat(x, model, idh, grade),tailor_and_concat(x.flip(dims=(2,)), model, idh, grade),\
#                                                             tailor_and_concat(x.flip(dims=(3,)), model, idh, grade),tailor_and_concat(x.flip(dims=(4,)), model, idh, grade),\
#                                                             tailor_and_concat(x.flip(dims=(2, 3)), model, idh, grade),tailor_and_concat(x.flip(dims=(2, 4)), model, idh, grade),\
#                                                             tailor_and_concat(x.flip(dims=(3, 4)), model, idh, grade),tailor_and_concat(x.flip(dims=(2, 3, 4)), model, idh, grade)
#
#
#         logit = F.softmax(TTA_1[0], 1)  # no flip
#         logit += F.softmax(TTA_2[0].flip(dims=(2,)), 1)  # flip H
#         logit += F.softmax(TTA_3[0].flip(dims=(3,)), 1)  # flip W
#         logit += F.softmax(TTA_4[0].flip(dims=(4,)), 1)  # flip D
#         logit += F.softmax(TTA_5[0].flip(dims=(2, 3)), 1)  # flip H, W
#         logit += F.softmax(TTA_6[0].flip(dims=(2, 4)), 1)  # flip H, D
#         logit += F.softmax(TTA_7[0].flip(dims=(3, 4)), 1)  # flip W, D
#         logit += F.softmax(TTA_8[0].flip(dims=(2, 3, 4)), 1)  # flip H, W, D
#         output = logit / 8.0   #TTA
#         idh_probs = []
#         grade_probs = []
#         for pred in [TTA_1, TTA_2, TTA_3, TTA_4, TTA_5, TTA_6, TTA_7, TTA_8]:
#             idh_probs.append(F.softmax(pred[1],1))
#             grade_probs.append(F.softmax(pred[2],1))
#
#         idh_pred = torch.mean(torch.stack(idh_probs),dim=0)
#         print("idh_pred:",idh_pred)
#         grade_pred = torch.mean(torch.stack(grade_probs),dim=0)
#         print("grade_pred:", grade_pred)
#
#         idh_pred_class = torch.argmax(idh_pred,dim=1) #类别概率值还需要比一个大小。
#         idh_class.append(idh_pred_class.item())
#         idh_prob1.append(idh_pred[0][1].item())
#         idh_conf.append(idh_pred[0][idh_pred_class.item()])  #如果是tensor，则[0]先是一整个tensor
#
#         print('id:',names[i],'IDH_truth:',idh.item(),'IDH_pred:',idh_pred_class.item())
#         #
#         grade_truth.append(grade.item())
#         grade_pred_class = torch.argmax(grade_pred, dim=1)
#         grade_class.append(grade_pred_class.item())
#         grade_prob1.append(grade_pred[0][1].item())
#         grade_conf.append(grade_pred[0][grade_pred_class.item()])
#         print('id:', names[i], 'grade_truth:', grade.item(), 'grade_pred:', grade_pred_class.item())
#         #
#         ids.append(names[i])
#         idh_truth.append(idh.item())
#         if not (idh_pred_class.item() == idh.item()):
#             idh_error_case.append({'id':names[i],'truth:':idh.item(),'pred':idh_pred_class.item()})
#         if not (grade_pred_class.item() == grade.item()):
#            grade_error_case.append({'id': names[i], 'truth:': grade.item(), 'pred': grade_pred_class.item()})
#
#         output = output[0, :, :H, :W, :T].cpu().detach().numpy()
#         output = output.argmax(0)   #分割结果的四张特征图的最大概率值所在标签（第二个维度-0/1/2/3）
#
#         name = str(i)
#         if names:
#             name = names[i]
#             msg += '{:>20}, '.format(name)
#
#         print(msg)
#
#         if savepath:
#             file_list = os.listdir(os.path.join('./upload/' + name + '/'))
#             image = nib.load(os.path.join('./upload/' + name + '/' + file_list[0]))
#             # .npy for further model ensemble
#             # .nii for directly model submission
#             assert save_format in ['npy', 'nii']
#             if save_format == 'npy':
#                 np.save(os.path.join(savepath, name + '_preds'), output)
#             if save_format == 'nii':
#                 # raise NotImplementedError
#                 # oname = os.path.join(savepath, name + '.nii.gz')
#                 seg_img = np.zeros(shape=(H, W, T), dtype=np.uint8)
#                 #
#                 seg_img[np.where(output == 1)] = 1
#                 seg_img[np.where(output == 2)] = 2
#                 seg_img[np.where(output == 3)] = 4
#                 #
#                 # print('NCR:', np.sum(seg_img == 1), ' | ED:', np.sum(seg_img == 2), ' | ET:', np.sum(seg_img == 4))
#                 WT_vol = np.sum((seg_img == 1) | (seg_img == 2) | (seg_img == 4))
#                 # # print('WT:', WT_vol , ' | TC:', np.sum((seg_img == 1) | (seg_img == 4)))
#                 #
#                 NCR_ratio.append(round(np.sum(seg_img == 1) / WT_vol, 2))
#                 ED_ratio.append(round(np.sum(seg_img == 2) / WT_vol, 2))
#                 ET_ratio.append(round(np.sum(seg_img == 4) / WT_vol, 2))
#
#                 print('NCR:', NCR_ratio, ' | ED:', ED_ratio, ' | ET:', ET_ratio)
#                 #
#                 # nib.save(nib.Nifti1Image(seg_img, affine=image.affine,header=image.header), oname)
#                 # print('Successfully save {}'.format(oname))
#
#                 if snapshot:
#                     """ --- grey figure---"""
#                     # Snapshot_img = np.zeros(shape=(H,W,T),dtype=np.uint8)
#                     # Snapshot_img[np.where(OriginalImg_output[1,:,:,:]==1)] = 64
#                     # Snapshot_img[np.where(OriginalImg_output[2,:,:,:]==1)] = 160
#                     # Snapshot_img[np.where(OriginalImg_output[3,:,:,:]==1)] = 255
#                     """ --- colorful figure--- """
#                     output = output.transpose(1, 0, 2)
#                     Snapshot_img = np.zeros(shape=(H, W, 3, T), dtype=np.uint8)
#                     Snapshot_img[:, :, 0, :][np.where(output == 1)] = 255
#                     Snapshot_img[:, :, 1, :][np.where(output == 2)] = 255
#                     Snapshot_img[:, :, 2, :][np.where(output == 4)] = 255
#
#                     for frame in range(T):   #每一个切片保存一张彩色图像
#                         if not os.path.exists(os.path.join(visual, name)):
#                             os.makedirs(os.path.join(visual, name))
#                         # scipy.misc.imsave(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])
#                         imageio.imwrite(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])
#     return {"idh_pred": idh_pred,
#             "grade_pred": grade_pred,
#             "idh_class": idh_pred_class,
#             "grade_class": grade_pred_class,
#             "NCR_ratio": NCR_ratio[0],
#             "ED_ratio": ED_ratio[0],
#             "ET_ratio": ET_ratio[0]}

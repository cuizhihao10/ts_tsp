import os
import time
import logging
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import numpy as np
import nibabel as nib
import imageio
from sklearn.metrics import roc_auc_score,accuracy_score,confusion_matrix,classification_report
import pandas as pd
import matplotlib.pyplot as plt
# import scikit-learn as sklearn
# from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
import nibabel as nib
import SimpleITK as sitk
import copy
from numpy import float64
from scipy.ndimage import label, find_objects

def caijian(a,b,c):
    (zstart, ystart, xstart), (zstop, ystop, xstop) = c.min(axis=-1), c.max(axis=-1) + 1
    roi_image = a[zstart:zstop, ystart:ystop, xstart :xstop ]
    roi_mask = b[zstart:zstop, ystart:ystop, xstart :xstop ]
    roi_image[roi_mask == 0] = 0
    # plt.imshow(roi_image[:,:,25],'gray')
    # plt.show()
    return roi_image

def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):

    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  ##获取原图size
    originSpacing = itkimage.GetSpacing()  ##获取原图spacing
    newSize = np.array(newSize,float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(int)   ##spacing格式转换

    resampler.SetReferenceImage(itkimage)   ##指定需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  ##得到重新采样后的图像

    return itkimgResampled

def largest_block_to_100(image):
    mask  = np.zeros_like(image)
    # 标记连通区域
    labeled_image, num_features = label(image)
    
    # 计算每个连通区域的体积
    volumes = np.array([np.sum(labeled_image == i) for i in range(1, num_features + 1)])
    
    # 找到体积最大的连通区域的标签
    largest_block_label = np.argmax(volumes) + 1
    
    # 将体积最大的连通区域中的像素值设置为100
    mask[labeled_image == largest_block_label] = 100
    
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

        x1_1, x2_1, x3_1,x4_1, encoder_output, weights_x = model['en'](temp[i])

        seg_output = model['seg'](x1_1, x2_1, x3_1,encoder_output)
        
        idh_out, grade_out, mgmt_out, pq_out = model['idh'](x4_1, encoder_output, idh, grade, mgmt ,pq) 

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
    mgmt_out = torch.mean(torch.stack(mgmt_temp), dim=0)
    pq_out = torch.mean(torch.stack(pq_temp), dim=0)
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
    from sklearn.metrics import roc_auc_score,accuracy_score,confusion_matrix,classification_report
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

    NCR_ratio =[]
    ED_ratio =[]
    ET_ratio = []
    dice_scores = []    
    dice_scores_label_2 = []
    dice_scores_label_1_4 = []
    postprocess = True

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

        TTA_1,TTA_2,TTA_3,TTA_4,TTA_5,TTA_6,TTA_7,TTA_8 = tailor_and_concat(x, model, idh, grade, mgmt, pq),tailor_and_concat(x.flip(dims=(2,)), model, idh, grade, mgmt, pq),\
                                                            tailor_and_concat(x.flip(dims=(3,)), model, idh, grade, mgmt, pq),tailor_and_concat(x.flip(dims=(4,)), model, idh, grade, mgmt, pq),\
                                                            tailor_and_concat(x.flip(dims=(2, 3)), model, idh, grade, mgmt, pq),tailor_and_concat(x.flip(dims=(2, 4)), model, idh, grade, mgmt, pq),\
                                                            tailor_and_concat(x.flip(dims=(3, 4)), model, idh, grade, mgmt, pq),tailor_and_concat(x.flip(dims=(2, 3, 4)), model, idh, grade, mgmt, pq)


        logit = F.softmax(TTA_1[0], 1)  # no flip
        logit += F.softmax(TTA_2[0].flip(dims=(2,)), 1)  # flip H
        logit += F.softmax(TTA_3[0].flip(dims=(3,)), 1)  # flip W
        logit += F.softmax(TTA_4[0].flip(dims=(4,)), 1)  # flip D
        logit += F.softmax(TTA_5[0].flip(dims=(2, 3)), 1)  # flip H, W
        logit += F.softmax(TTA_6[0].flip(dims=(2, 4)), 1)  # flip H, D
        logit += F.softmax(TTA_7[0].flip(dims=(3, 4)), 1)  # flip W, D
        logit += F.softmax(TTA_8[0].flip(dims=(2, 3, 4)), 1)  # flip H, W, D
        output = logit / 8.0   #TTA
        idh_probs = []
        grade_probs = []
        mgmt_probs = []
        pq_probs = []
        for pred in [TTA_1, TTA_2, TTA_3, TTA_4, TTA_5, TTA_6, TTA_7, TTA_8]:
            idh_probs.append(F.softmax(pred[1],1))
            grade_probs.append(F.softmax(pred[2],1))
            mgmt_probs.append(F.softmax(pred[3],1))
            pq_probs.append(F.softmax(pred[4],1))

        idh_pred = torch.mean(torch.stack(idh_probs),dim=0)
        print("idh_pred:",idh_pred)
        grade_pred = torch.mean(torch.stack(grade_probs),dim=0)
        print("grade_pred:", grade_pred)
        mgmt_pred = torch.mean(torch.stack(mgmt_probs),dim=0)
        print("mgmt_pred:", mgmt_pred)
        pq_pred = torch.mean(torch.stack(pq_probs),dim=0)
        print("pq_pred:", pq_pred)

        idh_pred_class = torch.argmax(idh_pred,dim=1) #类别概率值还需要比一个大小。
        idh_class.append(idh_pred_class.item())
        idh_prob1.append(idh_pred[0][1].item())
        idh_conf.append(idh_pred[0][idh_pred_class.item()])  #如果是tensor，则[0]先是一整个tensor

        print('id:',names[i],'IDH_truth:',idh.item(),'IDH_pred:',idh_pred_class.item())
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
            idh_error_case.append({'id':names[i],'truth:':idh.item(),'pred':idh_pred_class.item()})
        if not (grade_pred_class.item() == grade.item()):
           grade_error_case.append({'id': names[i], 'truth:': grade.item(), 'pred': grade_pred_class.item()})
        if not (mgmt_pred_class.item() == mgmt.item()):
            mgmt_error_case.append({'id': names[i], 'truth:': mgmt.item(), 'pred': mgmt_pred_class.item()})
        if not (pq_pred_class.item() == pq.item()):
            pq_error_case.append({'id': names[i], 'truth:': pq.item(), 'pred': pq_pred_class.item()})

        output = output[0, :, :H, :W, :T].cpu().detach().numpy()
        output = output.argmax(0)   #分割结果的四张特征图的最大概率值所在标签（第二个维度-0/1/2/3）

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
                oname = os.path.join(savepath, name, name + '.nii.gz')
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
                    itkimgResampled = resize_image_itk(itkimage, (128,128,128),
                                                resamplemethod= sitk.sitkLinear) ## resample使用线性插值
                    WT_img = sitk.GetArrayFromImage(itkimgResampled)           
                    # Calculate the pixel density
                    A, B, C = WT_img.shape
                    pixel_density = np.sum(WT_img) / (A * B * C)
                    print(f"The pixel density is: {pixel_density}")

                    mask = []
                    # Execute the functions as per the updated requirement
                    if pixel_density < 0.1:
                        print('I am entered.')
                        seg_img_1 = seg_img.copy()
                        pre_sum = np.count_nonzero(seg_img_1)
                        mask.append(largest_block_to_100(image))
                        print('pre_sum:', pre_sum)
                        seg_img_1[np.where(mask[-1] != 100)] = 0
                        post_sum = np.count_nonzero(seg_img_1)
                        print('post_sum——1:', post_sum)
                        if post_sum/pre_sum < 0.5:
                            while post_sum/pre_sum < 0.5:
                                print('do it again.')
                                image[np.where(mask[-1] == 100)] = 0
                                mask.append(largest_block_to_100(image))

                                seg_img_2 = seg_img.copy()
                                seg_img_3 = seg_img.copy()

                                for i in mask:
                                    seg_img_2[np.where(i == 100)] = 100   #todo 此处值需要在0-255之间，否则会无法统计

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
                ED_ratio.append(round(np.sum(seg_img == 2) / WT_vol, 2))
                ET_ratio.append(round(np.sum(seg_img == 4) / WT_vol, 2))
                
                # if torch.max(target) > 0.5:
                #     #todo 计算dice
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
                # # nib.save(nib.Nifti1Image(seg_img, affine=t1ce_image.affine,header=t1ce_image.header), oname)
                # # print('Successfully save {}'.format(oname))

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

                    for frame in range(T):   #每一个切片保存一张彩色图像
                        if not os.path.exists(os.path.join(visual, name)):
                            os.makedirs(os.path.join(visual, name))
                        # scipy.misc.imsave(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])
                        imageio.imwrite(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])

    return {"idh_pred": idh_pred,
            "grade_pred": grade_pred,
            "pq_pred": pq_pred,
            "idh_class": idh_pred_class,
            "grade_class": grade_pred_class,
            "pq_class": pq_pred_class,
            "NCR_ratio": NCR_ratio[0],
            "ED_ratio": ED_ratio[0],
            "ET_ratio": ET_ratio[0]}
    # print("-------------------------Save all labels and construct csv----------------------------------")
    # dice_df = pd.DataFrame({'Patient_ID': ids, 'DICE_Score_WT': dice_scores, 'DICE_Score_label_ED': dice_scores_label_2, 'DICE_Score_TC': dice_scores_label_1_4})
    # dice_df.to_csv(os.path.join('/public/home/hpc226511030/GMMAS-x/output', 'dice_scores.csv'), index=False)

    # print("--------------------------------IDH evaluation report---------------------------------------")
    
    # from sklearn.metrics import roc_auc_score,accuracy_score,confusion_matrix,classification_report
    # import pandas as pd
    
    # idh_conf = torch.tensor(idh_conf, device = 'cpu')
    # idh_class = torch.tensor(idh_class, device = 'cpu')
    # idh_truth = torch.tensor(idh_truth, device = 'cpu')

    # confusion = confusion_matrix(idh_truth,idh_class)

    # # #绘制彩色的混淆矩阵图
    # # plt.figure()
    # # plot_confusion_matrix(confusion, classes=["wild", "Mutant"], normalize=False,
    # #                       title='Confusion matrix, without normalization')
    # # plt.show()

    # classes=["wild", "Mutant"]
    # plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title("Confusion Matrix")
    # plt.colorbar()
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.yticks(tick_marks, classes)
    # #在格子内显示数值
    # thresh = confusion.max() / 2.
    # for i in range(confusion.shape[0]):
    #     for j in range(confusion.shape[1]):
    #         plt.text( j, i, format(confusion[i, j],'d'), horizontalalignment="center",
    #         color="white" if confusion[i, j] > thresh else "black")
    # plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.show()

    # print(confusion)
    # labels = [0, 1]
    # target_names = ["wild", "Mutant"]
    # print(classification_report(idh_truth, idh_class, labels=labels, target_names=target_names))
    # print("AUC:",roc_auc_score(idh_truth,idh_prob1))
    # print("Acc:", accuracy_score(idh_truth, idh_class))


    # #绘制AUC曲线图
    # fpr, tpr, thresholds = roc_curve(idh_truth, idh_prob1, pos_label=1)
    # np.savetxt('/public/home/hpc226511030/GMMAS-x/output/roc_dataIDH.txt', np.column_stack((fpr, tpr)), delimiter=',')
    # plt.figure()
    # plt.plot(fpr, tpr, linewidth=2, label='ROC of IDH')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.ylim(0, 1.05)
    # plt.xlim(0, 1.05)
    # plt.legend(loc=4)
    # plt.show()

    # if float64(np.sum(confusion)) != 0:
    #     accuracy = float64(confusion[0, 0] + confusion[1, 1]) / float64(np.sum(confusion))
    # print("Global Accuracy: " + str(accuracy))
    # specificity = 0
    # if float64(confusion[0, 0] + confusion[0, 1]) != 0:
    #     specificity = float64(confusion[0, 0]) / float64(confusion[0, 0] + confusion[0, 1])
    # print("Specificity: " + str(specificity))
    # sensitivity = 0
    # if float64(confusion[1, 1] + confusion[1, 0]) != 0:
    #     sensitivity = float64(confusion[1, 1]) / float64(confusion[1, 1] + confusion[1, 0])
    # print("Sensitivity: " + str(sensitivity))
    # precision = 0
    # if float64(confusion[1, 1] + confusion[0, 1]) != 0:
    #     precision = float64(confusion[1, 1]) / float64(confusion[1, 1] + confusion[0, 1])
    # print("Precision: " + str(precision))
    # print("-------------------------- error cases----------------------------------------")
    # for case in idh_error_case:
    #     print(case)

    # print("--------------------------------grade evaluation report---------------------------------------")

    # from sklearn.metrics import roc_auc_score,accuracy_score,confusion_matrix,classification_report
    # import pandas as pd

    # grade_conf = torch.tensor(grade_conf, device = 'cpu')
    # grade_class = torch.tensor(grade_class, device = 'cpu')
    # grade_truth = torch.tensor(grade_truth, device = 'cpu')

    # confusion = confusion_matrix(grade_truth,grade_class)
    # print(confusion)
    # #绘制彩色的混淆矩阵图
    # # plt.figure()
    # # plot_confusion_matrix(confusion, classes=["LGG", "HGG"], normalize=False,
    # #                       title='Confusion matrix, without normalization')
    # # plt.show()

    # labels = [0, 1]
    # target_names = ["LGG", "HGG"]
    # print(classification_report(grade_truth, grade_class, labels=labels, target_names=target_names))
    # print("AUC:",roc_auc_score(grade_truth,grade_prob1))
    # print("Acc:", accuracy_score(grade_truth, grade_class))
    # #绘制AUC曲线图
    # fpr, tpr, thresholds = roc_curve(grade_truth, grade_prob1, pos_label=1)
    # np.savetxt('/public/home/hpc226511030/GMMAS-x/output/roc_dataGrade.txt', np.column_stack((fpr, tpr)), delimiter=',')
    # plt.figure()
    # plt.plot(fpr, tpr, linewidth=2, label='ROC of Grade')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.ylim(0, 1.05)
    # plt.xlim(0, 1.05)
    # plt.legend(loc=4)
    # plt.show()

    # if float64(np.sum(confusion)) != 0:
    #     accuracy = float64(confusion[0, 0] + confusion[1, 1]) / float64(np.sum(confusion))
    # print("Global Accuracy: " + str(accuracy))
    # specificity = 0
    # if float64(confusion[0, 0] + confusion[0, 1]) != 0:
    #     specificity = float64(confusion[0, 0]) / float64(confusion[0, 0] + confusion[0, 1])
    # print("Specificity: " + str(specificity))
    # sensitivity = 0
    # if float64(confusion[1, 1] + confusion[1, 0]) != 0:
    #     sensitivity = float64(confusion[1, 1]) / float64(confusion[1, 1] + confusion[1, 0])
    # print("Sensitivity: " + str(sensitivity))
    # precision = 0
    # if float64(confusion[1, 1] + confusion[0, 1]) != 0:
    #     precision = float64(confusion[1, 1]) / float64(confusion[1, 1] + confusion[0, 1])
    # print("Precision: " + str(precision))
    # print("-------------------------- error cases----------------------------------------")
    # for case in grade_error_case:
    #     print(case)

    # print("--------------------------------mgmt evaluation report---------------------------------------")
    
    # from sklearn.metrics import roc_auc_score,accuracy_score,confusion_matrix,classification_report
    # import pandas as pd
    # mgmt_conf = torch.tensor(mgmt_conf, device = 'cpu')
    # mgmt_class = torch.tensor(mgmt_class, device = 'cpu')
    # mgmt_truth = torch.tensor(mgmt_truth, device = 'cpu')

    # confusion = confusion_matrix(mgmt_truth,mgmt_class)
    # print(confusion)
    #     #绘制彩色的混淆矩阵图
    # # plt.figure()
    # # plot_confusion_matrix(confusion, classes=["unmethylated", "methylated"], normalize=False,
    # #                       title='Confusion matrix, without normalization')
    # # plt.show()

    # labels = [0, 1]
    # target_names = ["unmethylated", "methylated"]
    # print(classification_report(mgmt_truth, mgmt_class, labels=labels, target_names=target_names))
    # print("AUC:",roc_auc_score(mgmt_truth,mgmt_prob1))
    # print("Acc:", accuracy_score(mgmt_truth, mgmt_class))
    # #绘制AUC曲线图
    # fpr, tpr, thresholds = roc_curve(mgmt_truth, mgmt_prob1, pos_label=1)
    # np.savetxt('/public/home/hpc226511030/GMMAS-x/output/roc_datamgmt.txt', np.column_stack((fpr, tpr)), delimiter=',')
    # plt.figure()
    # plt.plot(fpr, tpr, linewidth=2, label='ROC of MGMT')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.ylim(0, 1.05)
    # plt.xlim(0, 1.05)
    # plt.legend(loc=4)
    # plt.show()

    # if float64(np.sum(confusion)) != 0:
    #     accuracy = float64(confusion[0, 0] + confusion[1, 1]) / float64(np.sum(confusion))
    # print("Global Accuracy: " + str(accuracy))
    # specificity = 0
    # if float64(confusion[0, 0] + confusion[0, 1]) != 0:
    #     specificity = float64(confusion[0, 0]) / float64(confusion[0, 0] + confusion[0, 1])
    # print("Specificity: " + str(specificity))
    # sensitivity = 0
    # if float64(confusion[1, 1] + confusion[1, 0]) != 0:
    #     sensitivity = float64(confusion[1, 1]) / float64(confusion[1, 1] + confusion[1, 0])
    # print("Sensitivity: " + str(sensitivity))
    # precision = 0
    # if float64(confusion[1, 1] + confusion[0, 1]) != 0:
    #     precision = float64(confusion[1, 1]) / float64(confusion[1, 1] + confusion[0, 1])
    # print("Precision: " + str(precision))
    # print("-------------------------- error cases----------------------------------------")
    # for case in mgmt_error_case:
    #     print(case)

    # print("--------------------------------pq evaluation report---------------------------------------")
        
    # from sklearn.metrics import roc_auc_score,accuracy_score,confusion_matrix,classification_report
    # import pandas as pd
    # pq_conf = torch.tensor(pq_conf, device = 'cpu')
    # pq_class = torch.tensor(pq_class, device = 'cpu')
    # pq_truth = torch.tensor(pq_truth, device = 'cpu')

    # confusion = confusion_matrix(pq_truth,pq_class)
    # print(confusion)
    #     #绘制彩色的混淆矩阵图
    # # plt.figure()
    # # plot_confusion_matrix(confusion, classes=["non-paired", "paired"], normalize=False,
    # #                       title='Confusion matrix, without normalization')
    # # plt.show()

    # labels = [0, 1]
    # target_names = ["non-paired", "paired"]
    # print(classification_report(pq_truth, pq_class, labels=labels, target_names=target_names))
    # print("AUC:",roc_auc_score(pq_truth,pq_prob1))
    # print("Acc:", accuracy_score(pq_truth, pq_class))
    # #绘制AUC曲线图
    # fpr, tpr, thresholds = roc_curve(pq_truth, pq_prob1, pos_label=1)
    # plt.figure()
    # plt.plot(fpr, tpr, linewidth=2, label='ROC of PQ')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.ylim(0, 1.05)
    # plt.xlim(0, 1.05)
    # plt.legend(loc=4)
    # plt.show()
    
    # if float64(np.sum(confusion)) != 0:
    #     accuracy = float64(confusion[0, 0] + confusion[1, 1]) / float64(np.sum(confusion))
    # print("Global Accuracy: " + str(accuracy))
    # specificity = 0
    # if float64(confusion[0, 0] + confusion[0, 1]) != 0:
    #     specificity = float64(confusion[0, 0]) / float64(confusion[0, 0] + confusion[0, 1])
    # print("Specificity: " + str(specificity))
    # sensitivity = 0
    # if float64(confusion[1, 1] + confusion[1, 0]) != 0:
    #     sensitivity = float64(confusion[1, 1]) / float64(confusion[1, 1] + confusion[1, 0])
    # print("Sensitivity: " + str(sensitivity))
    # precision = 0
    # if float64(confusion[1, 1] + confusion[0, 1]) != 0:
    #     precision = float64(confusion[1, 1]) / float64(confusion[1, 1] + confusion[0, 1])
    # print("Precision: " + str(precision))
    # print("-------------------------- error cases----------------------------------------")
    # for case in pq_error_case:
    #     print(case)
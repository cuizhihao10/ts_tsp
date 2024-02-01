import base64
from PIL import Image, ImageFilter
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models import UploadFile, SegmentationFile, ClassificationFile
from extension import db
from io import BytesIO
import config
import json
import os
import time
from datetime import datetime
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
import torch.optim
from torch.utils.data import DataLoader
from inference_model.BraTS_IDH import BraTS
from inference_model.predict import validate_softmax
from inference_model.TransBraTS.TransBraTS_skipconnection import TransBraTS, Decoder_modual, IDH_network
from utils.niiTopng import nii2png



def get_folder_size(folder_path):
    total_size = 0
    for path, dirs, files in os.walk(folder_path):
        for f in files:
            fp = os.path.join(path, f)
            total_size += os.path.getsize(fp)
    return total_size


def main(upload_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    model = TransBraTS(dataset='brats', _conv_repr=True, _pe_type="learned")
    seg_model = Decoder_modual()
    IDH_model = IDH_network()
    model = torch.nn.DataParallel(model).cuda()
    seg_model = torch.nn.DataParallel(seg_model).cuda()
    IDH_model = torch.nn.DataParallel(IDH_model).cuda()
    dict_model = {'en': model, 'seg': seg_model, 'idh': IDH_model}
    load_file = Path('inference_model/model_epoch_3.pth')
    if os.path.exists(load_file):
        checkpoint = torch.load(load_file)
        dict_model['en'].load_state_dict(checkpoint['en_state_dict'])
        dict_model['seg'].load_state_dict(checkpoint['seg_state_dict'])
        dict_model['idh'].load_state_dict(checkpoint['idh_state_dict'])
    else:
        print('There is no resume file to load!')
    valid_root = os.path.join('./upload/', upload_path)
    valid_set = BraTS(valid_root, mode='test')
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    submission = os.path.join('inference_model/output', 'submission')
    visual = os.path.join('inference_model/output', 'visualization')

    if not os.path.exists(submission):
        os.makedirs(submission)
    if not os.path.exists(visual):
        os.makedirs(visual)

    start_time = time.time()

    with torch.no_grad():
        result = validate_softmax(valid_loader=valid_loader,
                                               model=dict_model,
                                               savepath=submission,
                                               visual=visual,
                                               names=valid_set.names,
                                               save_format='nii',
                                               snapshot=True
                                               )
    classific_result = {
        'idh_pred': result['idh_pred'],
        'idh_class': result['idh_class'],
        'pq_pred': result['pq_pred'],
        'pq_class': result['pq_class'],
        'grade_pred': result['grade_pred'],
        'grade_class': result['grade_class']
    }
    end_time = time.time()
    full_test_time = (end_time - start_time) / 60
    average_time = full_test_time / len(valid_set)
    visual_path = os.path.join('inference_model/output/visualization/', upload_path)[:-1]
    print('{:.2f} minutes!'.format(average_time))

    return {'classific_result': classific_result,
            'visual_path': visual_path,
            'visual_images_size': round(get_folder_size(visual_path) / 1024 / 1024, 2),
            'type': 'segmentation',
            'NCR_ratio': result['NCR_ratio'],
            'ED_ratio': result['ED_ratio'],
            'ET_ratio': result['ET_ratio'],
            'nowtime': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


def show_images(ori_img_path, seg_img_path):
    original_images = {}
    sequence_list = []
    original_path = os.path.join('OriginalImg_output', ori_img_path)
    original_sequence_files = os.listdir(original_path)
    for sequence in original_sequence_files:
        sequence_name = sequence.split('_')[-1].split('.')[0]
        sequence_list.append(sequence_name)
        original_images[sequence_name] = []
        original_images_list = os.listdir(os.path.join(original_path, sequence))
        original_images_list.sort(key=lambda x: int(x.split('.')[0]))
        for i in range(len(original_images_list)):
            # original_images.append('http://127.0.0.1:5000/OriginalImg_output/' + ori_img_path + '/' + sequence + '/' +
            #                        original_images_list[i])
            original_image = Image.open(os.path.join(original_path, sequence, original_images_list[i]))
            # original_image = original_image.filter(ImageFilter.SMOOTH)
            original_img_io = BytesIO()
            original_image.save(original_img_io, 'PNG')
            original_img_str = base64.b64encode(original_img_io.getvalue())
            original_images[sequence_name].append('data:image/png;base64,' + str(original_img_str).split("'")[1])

    seg_images = []
    seg_path = os.path.join('inference_model/output/visualization/', seg_img_path)
    # seg_path = os.path.join(os.path.join('./inference_model/output/visualization/', '111'))
    seg_images_list = os.listdir(seg_path)
    seg_images_list.sort(key=lambda x: int(x.split('.')[0]))
    for i in range(len(seg_images_list)):
        # seg_images.append(
        #     'http://127.0.0.1:5000/inference_model/output/visualization/' + seg_img_path + '/' + seg_images_list[i])
        seg_image = Image.open(os.path.join(seg_path, seg_images_list[i]))
        seg_img_io = BytesIO()
        seg_image.save(seg_img_io, 'PNG')
        seg_img_str = base64.b64encode(seg_img_io.getvalue())
        seg_images.append('data:image/png;base64,' + str(seg_img_str).split("'")[1])
    return original_images, seg_images, original_path, sequence_list


bp = Blueprint("segmentation", __name__, url_prefix="/segmentation")


@bp.route("/test/", methods=["POST"])
def test():
    return jsonify({"code": 200, "msg": "test success"})


@bp.route("/upload/", methods=["POST"])
@jwt_required()
def post_file():
    files = request.files.getlist("files")
    filename = request.form.get("task_name")
    identity = get_jwt_identity()
    # file_type = request.form.get("task_type")
    upload_time = request.form.get("upload_time")
    print(request.form)
    if UploadFile.query.filter_by(filename=filename).first():
        return jsonify({"code": 500, "msg": "upload failed", "error_message": "Filename already exists"})
    else:
        os.mkdir(os.path.join(config.UPLOAD_FOLDER, filename))
    for file in files:
        file.save(os.path.join(config.UPLOAD_FOLDER, filename, file.filename))  # 保存文件
    file_size = round(get_folder_size(os.path.join(config.UPLOAD_FOLDER, filename)) / 1024 / 1024, 2)  # 文件大小
    # 保存文件信息到数据库
    upload_file = UploadFile(filename=filename, sub_files=",".join([file.filename for file in files]),
                             size=file_size, path=os.path.join(config.UPLOAD_FOLDER, filename),
                             upload_time=upload_time, author_id=identity)
    db.session.add(upload_file)
    db.session.commit()
    return jsonify({"code": 200, "msg": "upload success"})


@bp.route("/modeling/", methods=["POST"])
@jwt_required()
def modeling():
    original_num = {}
    task_name = request.json.get("task_name")
    upload_path = task_name + '/'
    nii2png(os.path.join(config.UPLOAD_FOLDER, upload_path))
    output_info = main(upload_path)
    seg_result = json.dumps(
        {"NCR_ratio": "%.2f%%" % (output_info['NCR_ratio'] * 100),
            "ED_ratio": "%.2f%%" % (output_info['ED_ratio'] * 100),
            "ET_ratio": "%.2f%%" % (output_info['ET_ratio'] * 100),
        }
    )
    classific_result = json.dumps(
        {"idh_wild": "%.2f%%" % (output_info['classific_result']['idh_pred'][0][0].item() * 100),
            "idh_mutant": "%.2f%%" % (output_info['classific_result']['idh_pred'][0][1].item() * 100),
            "idh_class": "%s" % (output_info['classific_result']['idh_class'][0].item() == 0 and 'Wild' or 'Mutant'),
            "1p/19q intac": "%.2f%%" % (output_info['classific_result']['pq_pred'][0][0].item() * 100),
            "1p/19q codel": "%.2f%%" % (output_info['classific_result']['pq_pred'][0][1].item() * 100),
            "pq_class": "%s" % (output_info['classific_result']['pq_class'][0].item() == 0 and '1p/19q intac' or '1p/19q codel'),
            "grade_LGG": "%.2f%%" % (output_info['classific_result']['grade_pred'][0][0].item() * 100),
            "grade_HGG": "%.2f%%" % (output_info['classific_result']['grade_pred'][0][1].item() * 100),
            "grade_class": "%s" % (output_info['classific_result']['grade_class'][0].item() == 0 and 'LGG' or 'GBM'),
         })
    original_images, seg_images, original_path, original_sequence_list = show_images(task_name, task_name)
    for key in original_images.keys():
        original_num[key] = len(original_images[key]) - 1
    segmentationfile = SegmentationFile(filename=task_name,
                                        type='segmentation',
                                        size=output_info['visual_images_size'],
                                        path=output_info['visual_path'],
                                        original_images_sequence=','.join(original_sequence_list),
                                        original_images_path=original_path,
                                        segmentation_result=seg_result,
                                        segmentation_time=output_info['nowtime'],
                                        author_id=get_jwt_identity())
    classificationfile = ClassificationFile(filename=task_name,
                                            type='classification',
                                            size=output_info['visual_images_size'],
                                            path=output_info['visual_path'],
                                            original_images_sequence=','.join(original_sequence_list),
                                            original_images_path=original_path,
                                            classification_result=classific_result,
                                            classification_time=output_info['nowtime'],
                                            author_id=get_jwt_identity())
    db.session.add(segmentationfile)
    db.session.add(classificationfile)
    db.session.commit()
    return jsonify({"code": 200, "msg": "modeling success", "data": {"original_images": original_images,
                                                                     "seg_images": seg_images,
                                                                     "original_num": original_num,
                                                                     "seg_num": len(seg_images) - 1,
                                                                     "seg_result": seg_result,
                                                                     "classific_result": classific_result}})

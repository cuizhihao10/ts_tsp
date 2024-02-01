import os
import base64
import shutil
from extension import db
from io import BytesIO
from PIL import Image, ImageFilter
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models import UploadFile, SegmentationFile, ClassificationFile
from sqlalchemy import cast, String

bp = Blueprint("history", __name__, url_prefix="/history")


def show_images(ori_img_path, seg_img_path):
    original_images = {}
    sequence_list = []
    original_sequence_files = os.listdir(ori_img_path)
    for sequence in original_sequence_files:
        sequence_name = sequence.split('_')[-1].split('.')[0]
        sequence_list.append(sequence_name)
        original_images[sequence_name] = []
        original_images_list = os.listdir(os.path.join(ori_img_path, sequence))
        original_images_list.sort(key=lambda x: int(x.split('.')[0]))
        for i in range(len(original_images_list)):
            original_image = Image.open(os.path.join(ori_img_path, sequence, original_images_list[i]))
            # original_image = original_image.convert("L")
            # original_image = original_image.filter(ImageFilter.GaussianBlur(radius=2))
            # edges_image = original_image.filter(ImageFilter.FIND_EDGES)
            # original_image = original_image.resize(original_image.size, Image.BILINEAR)
            original_img_io = BytesIO()
            original_image.save(original_img_io, 'PNG')
            original_img_str = base64.b64encode(original_img_io.getvalue())
            original_images[sequence_name].append('data:image/png;base64,' + str(original_img_str).split("'")[1])

    seg_images = []
    seg_images_list = os.listdir(seg_img_path)
    seg_images_list.sort(key=lambda x: int(x.split('.')[0]))
    print(seg_images_list)
    for i in range(len(seg_images_list)):
        seg_image = Image.open(
            os.path.join(os.path.join(os.path.join(seg_img_path, seg_images_list[i]))))
        seg_img_io = BytesIO()
        seg_image.save(seg_img_io, 'PNG')
        seg_img_str = base64.b64encode(seg_img_io.getvalue())
        seg_images.append('data:image/png;base64,' + str(seg_img_str).split("'")[1])
    return original_images, seg_images, sequence_list


@bp.route("/segmentation/", methods=["POST"])
@jwt_required()
def show_segmentation_history():
    post_data = request.get_json()
    page = post_data.get('page')
    page_size = post_data.get('page_size')
    identity = post_data.get("userId")
    SegmentationFile_page = (SegmentationFile.query.filter_by(author_id=identity)
                             .order_by(SegmentationFile.segmentation_time.desc())
                             .paginate(page=page, per_page=page_size))
    segmentation_task_total = SegmentationFile.query.filter_by(author_id=identity).count()
    segmentation_history_list = []
    for history in SegmentationFile_page:
        segmentation_history_list.append({"filename": history.filename, "type": history.type,
                                          "size": history.size, "path": history.path,
                                          "original_images_sequence": history.original_images_sequence,
                                          "original_images_path": history.original_images_path,
                                          "segmentation_time": history.segmentation_time})
    return jsonify({"code": 200, "msg": "get segmentation history success", "data": {
        "page_list": segmentation_history_list,
        "total": segmentation_task_total
    }})


@bp.route("/search_autocomplete/", methods=["POST"])
# @jwt_required()
def search_autocomplete():
    post_data = request.get_json()
    identity = post_data.get("userId")
    SegmentationFile_all = (SegmentationFile.query.filter_by(author_id=identity)).all()
    SegmentationFile_count = (SegmentationFile.query.filter_by(author_id=identity)).count()
    search_autocomplete_list = []
    for history in SegmentationFile_all:
        search_autocomplete_list.append({"task_name": history.filename,
                                          "create_time": history.segmentation_time})
    return jsonify({"code": 200, "msg": "get segmentation history success", "data": {
        "task_list": search_autocomplete_list,
        "total": SegmentationFile_count
    }})


@bp.route("/segmentation_search/", methods=["POST"])
@jwt_required()
def search_segmentation_history():
    post_data = request.get_json()
    select = post_data.get("select")
    page = post_data.get('page')
    page_size = post_data.get('page_size')
    identity = post_data.get("userId")
    try:
        if select:
            if select == "task_name":
                task_name = post_data.get("state")
                SegmentationFile_page = (SegmentationFile.query.filter_by(author_id=identity, filename=task_name)
                                         .order_by(SegmentationFile.segmentation_time.desc())
                                         .paginate(page=page, per_page=page_size))
                segmentation_task_total = SegmentationFile.query.filter_by(author_id=identity,
                                                                           filename=task_name).count()
                if segmentation_task_total == 0:
                    return jsonify({"code": 300, "msg": "no such task"})
                segmentation_history_list = []
                for history in SegmentationFile_page:
                    segmentation_history_list.append({"filename": history.filename, "type": history.type,
                                                      "size": history.size, "path": history.path,
                                                      "original_images_sequence": history.original_images_sequence,
                                                      "original_images_path": history.original_images_path,
                                                      "segmentation_time": history.segmentation_time})
                return jsonify({"code": 200, "msg": "get segmentation history success", "data": {
                    "page_list": segmentation_history_list,
                    "total": segmentation_task_total
                }})
            elif select == "create_time":
                create_time = post_data.get("state")
                print(type(create_time))
                SegmentationFile_page = (
                    SegmentationFile.query.filter(
                        SegmentationFile.author_id == identity,
                        cast(SegmentationFile.segmentation_time, String).like(f"%{create_time}%")
                    )
                    .order_by(SegmentationFile.segmentation_time.desc())
                    .paginate(page=page, per_page=page_size)
                )
                segmentation_task_total = SegmentationFile.query.filter(SegmentationFile.author_id == identity,
                                                                        SegmentationFile.segmentation_time.like(
                                                                            f"%{create_time}%")).count()
                print(segmentation_task_total)
                if segmentation_task_total == 0:
                    return jsonify({"code": 300, "msg": "no such task"})
                segmentation_history_list = []
                for history in SegmentationFile_page:
                    segmentation_history_list.append({"filename": history.filename, "type": history.type,
                                                      "size": history.size, "path": history.path,
                                                      "original_images_sequence": history.original_images_sequence,
                                                      "original_images_path": history.original_images_path,
                                                      "segmentation_time": history.segmentation_time})
                return jsonify({"code": 200, "msg": "get segmentation history success", "data": {
                    "page_list": segmentation_history_list,
                    "total": segmentation_task_total
                }})
        else:
            return jsonify({"code": 300, "msg": "please select search type"})
    except Exception as e:
        return jsonify({"code": 500, "msg": "search failed"})


@bp.route("/segmentation_detail/", methods=["GET"])
@jwt_required()
def show_segmentation_taskdetail():
    original_num = {}
    identity = request.args.get("userId")
    task_name = request.args.get("task_name")
    segmentation_task = SegmentationFile.query.filter_by(author_id=identity, filename=task_name).first()
    path = segmentation_task.path
    original_images_path = segmentation_task.original_images_path
    segmentation_result = segmentation_task.segmentation_result
    original_images, seg_images, original_sequence_list = show_images(original_images_path, path)
    for key in original_images.keys():
        original_num[key] = len(original_images[key]) - 1
    return jsonify({"code": 200, "msg": "modeling success", "data": {"original_images": original_images,
                                                                     "seg_images": seg_images,
                                                                     "original_num": original_num,
                                                                     "seg_num": len(seg_images) - 1,
                                                                     "original_sequence_list": original_sequence_list,
                                                                     "segmentation_result": segmentation_result}})


@bp.route("/segmentation_delete/", methods=["POST"])
@jwt_required()
def delete_segmentation_task():
    post_data = request.get_json()
    task_name = post_data.get("task_name")
    identity = post_data.get("userId")
    upload_task = UploadFile.query.filter_by(author_id=identity, filename=task_name).first()
    segmentation_task = (SegmentationFile.query.filter_by(author_id=identity, filename=task_name).first())
    if segmentation_task:
        shutil.rmtree(segmentation_task.path)
        shutil.rmtree(segmentation_task.original_images_path)
        shutil.rmtree(upload_task.path)
        db.session.delete(segmentation_task)
        db.session.delete(upload_task)
        db.session.commit()
        return jsonify({"code": 200, "msg": "delete success"})
    else:
        return jsonify({"code": 500, "msg": "delete failed"})


@bp.route("/classification/", methods=["POST"])
@jwt_required()
def show_classification_history():
    post_data = request.get_json()
    page = post_data.get('page')
    page_size = post_data.get('page_size')
    identity = post_data.get("userId")
    ClassificationFile_page = (ClassificationFile.query.order_by(ClassificationFile.classification_time.desc())
                               .paginate(page=page, per_page=page_size))
    classification_task_total = ClassificationFile.query.filter_by(author_id=identity).count()
    classification_history_list = []
    for history in ClassificationFile_page:
        classification_history_list.append({"filename": history.filename, "type": history.type,
                                            "size": history.size, "path": history.path,
                                            "original_images_sequence": history.original_images_sequence,
                                            "original_images_path": history.original_images_path,
                                            "classification_time": history.classification_time,
                                            "classification_result": history.classification_result})
    return jsonify({"code": 200,
                    "msg": "get classification history success",
                    "data":
                        {
                            "page_list": classification_history_list,
                            "total": classification_task_total
                        }
                    })


@bp.route("/classification_detail/", methods=["GET"])
@jwt_required()
def show_classification_taskdetail():
    original_num = {}
    identity = request.args.get("userId")
    task_name = request.args.get("task_name")
    classification_task = ClassificationFile.query.filter_by(author_id=identity, filename=task_name).first()
    path = classification_task.path
    original_images_path = classification_task.original_images_path
    original_images, seg_images, original_sequence_list = show_images(original_images_path, path)
    for key in original_images.keys():
        original_num[key] = len(original_images[key]) - 1
    return jsonify({"code": 200, "msg": "modeling success", "data": {"original_images": original_images,
                                                                     "seg_images": seg_images,
                                                                     "original_num": original_num,
                                                                     "seg_num": len(seg_images) - 1,
                                                                     "original_sequence_list": original_sequence_list}})


@bp.route("/classification_delete/", methods=["POST"])
@jwt_required()
def delete_classification_task():
    post_data = request.get_json()
    task_name = post_data.get("task_name")
    identity = post_data.get("userId")
    upload_task = UploadFile.query.filter_by(author_id=identity, filename=task_name).first()
    classification_task = (ClassificationFile.query.filter_by(author_id=identity, filename=task_name).first())
    if classification_task:
        shutil.rmtree(classification_task.path)
        shutil.rmtree(classification_task.original_images_path)
        shutil.rmtree(upload_task.path)
        db.session.delete(classification_task)
        db.session.delete(upload_task)
        db.session.commit()
        return jsonify({"code": 200, "msg": "delete success"})
    else:
        return jsonify({"code": 500, "msg": "delete failed"})

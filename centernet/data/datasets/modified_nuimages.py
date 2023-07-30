from detectron2.data.datasets.register_coco import register_coco_instances
import os

categories = [
    {'id': 1, 'name': 'car'},
    {'id': 2, 'name': 'truck'},
    {'id': 3, 'name': 'construction_vehicle'},
    {'id': 4, 'name': 'bus'}, 
    {'id': 5, 'name': 'trailer'},
    {'id': 6, 'name': 'emergency'}, 
    {'id': 7, 'name': 'motorcycle'}, 
    {'id': 8, 'name': 'bicycle'}, 
    {'id': 9, 'name': 'adult'}, 
    {'id': 10, 'name': 'child'},
    {'id': 11, 'name': 'police_officer'}, 
    {'id': 12, 'name': 'construction_worker'},
    {'id': 13, 'name': 'personal_mobility'}, 
    {'id': 14, 'name': 'wheelchair'},
    {'id': 15, 'name': 'stroller'}, 
    {'id': 16, 'name': 'pushable_pullable'},
    {'id': 17, 'name': 'barrier'}, 
    {'id': 18, 'name': 'traffic_cone'},
    {'id': 19, 'name': 'debris'}
]
def _get_builtin_metadata():
    id_to_name = {x['id']: x['name'] for x in categories}
    thing_dataset_id_to_contiguous_id = {i: i for i in range(len(categories))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}

_PREDEFINED_SPLITS = {
    "modified_nuimages_train": ("/home/anishmad/msr_thesis/glip/DATASET/nuimages/images", "/home/anishmad/msr_thesis/glip/DATASET/nuimages/modified_annotations/nuimages_v1.0-train.json"),
    "modified_nuimages_val": ("/home/anishmad/msr_thesis/glip/DATASET/nuimages/images", "/home/anishmad/msr_thesis/glip/DATASET/nuimages/modified_annotations/nuimages_v1.0-val.json"),
    "modified_nuimages_mini": ("/home/anishmad/msr_thesis/glip/DATASET/nuimages/images", "/home/anishmad/msr_thesis/glip/DATASET/nuimages/modified_annotations/nuimages_dummy_v1.0-val.json"),
}

for key, (image_root, json_file) in _PREDEFINED_SPLITS.items():
    register_coco_instances(
        key,
        _get_builtin_metadata(),
        json_file,
        image_root,
    )

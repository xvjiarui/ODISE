from .register_seginw import _CATEGORIES, get_openseg_labels

from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import get_detection_dataset_dicts
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetMapper

from odise.modeling.wrapper.pano_wrapper import OpenPanopticInference
from odise.data import build_d2_test_dataloader

from odise.evaluation.d2_evaluator import InstanceSegEvaluator

tasks = dict()
for cat in _CATEGORIES:

    dataset_name = f"seginw_{cat}_val"
    seginw_open_eval = OmegaConf.create()
    seginw_open_eval.loader = L(build_d2_test_dataloader)(
        dataset=L(get_detection_dataset_dicts)(
            names=dataset_name, filter_empty=False
        ),
        mapper=L(DatasetMapper)(
            is_train=False,
            augmentations=[
                L(T.ResizeShortestEdge)(
                    short_edge_length=1024,
                    max_size=2560,
                    sample_style="choice",
                ),
            ],
            image_format="RGB",
        ),
        local_batch_size=1,
        num_workers=1,
    )

    seginw_open_eval.wrapper = L(OpenPanopticInference)(
        labels=L(get_openseg_labels)(dataset_name),
        metadata=L(MetadataCatalog.get)(name=dataset_name),
        semantic_on=False,
        instance_on=True,
        panoptic_on=False,
    )

    seginw_open_eval.evaluator = [
        L(InstanceSegEvaluator)(
            dataset_name="${...loader.dataset.names}",
            tasks=("segm",),
        ),
    ]
    tasks[f"eval_{dataset_name.lower()}"] = seginw_open_eval
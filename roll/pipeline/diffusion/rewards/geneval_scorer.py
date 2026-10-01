import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps


class GenevalScorer:
    def __init__(
        self,
        detector_ckpt_path: str,
        detector_config_path: str,
        device: str = "cuda",
        clip_ckpt_path: Optional[str] = None,
        object_names_path: Optional[str] = None,
        threshold: float = 0.3,
        counting_threshold: float = 0.9,
        max_objects: int = 16,
        # Mask2Former is query-based and intentionally NMS-free, but at inference
        # several queries can lock onto the same object with different scores
        # (observed: two handbag boxes at IoU 0.94, the lower one fully contained).
        # Upstream GenEval defaults to 1.0, which short-circuits the dedup loop
        # below and lets such duplicates inflate object counts, so strict
        # "exactly N objects" checks fail on correct images. 0.9 is deliberately
        # conservative: it drops near-identical boxes only, keeping genuinely
        # overlapping same-class instances that counting prompts depend on.
        nms_threshold: float = 0.9,
        position_threshold: float = 0.1,
    ):
        self.device = device
        self.threshold = threshold
        self.counting_threshold = counting_threshold
        self.max_objects = max_objects
        self.nms_threshold = nms_threshold
        self.position_threshold = position_threshold

        self._load_dependencies()
        self.object_detector = self.init_detector(detector_config_path, detector_ckpt_path, device=device)

        clip_arch = "ViT-L-14"
        clip_pretrained = clip_ckpt_path or "openai"
        self.clip_model, _, self.transform = self.open_clip.create_model_and_transforms(
            clip_arch,
            pretrained=clip_pretrained,
            device=device,
        )
        self.tokenizer = self.open_clip.get_tokenizer(clip_arch)

        names_path = Path(object_names_path) if object_names_path else Path(__file__).with_name("object_names.txt")
        self.classnames = [line.strip() for line in names_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.colors = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
        self.color_classifiers: Dict[str, torch.Tensor] = {}

    def _load_dependencies(self):
        import mmdet
        import open_clip
        from clip_benchmark.metrics import zeroshot_classification as zsc
        from mmdet.apis import inference_detector, init_detector

        self.mmdet = mmdet
        zsc.tqdm = lambda it, *args, **kwargs: it
        self.open_clip = open_clip
        self.zsc = zsc
        self.inference_detector = inference_detector
        self.init_detector = init_detector

    class _ImageCrops(torch.utils.data.Dataset):
        def __init__(self, image: Image.Image, objects, transform):
            self._image = image.convert("RGB")
            self._blank = Image.new("RGB", image.size, color="#999")
            self._objects = objects
            self._transform = transform

        def __len__(self):
            return len(self._objects)

        def __getitem__(self, index):
            box, mask = self._objects[index]
            image = self._image
            if mask is not None:
                image = Image.composite(self._image, self._blank, Image.fromarray(mask))
            image = image.crop(box[:4])
            return (self._transform(image), 0)

    def _color_classification(self, image: Image.Image, bboxes, classname: str) -> List[str]:
        if classname not in self.color_classifiers:
            self.color_classifiers[classname] = self.zsc.zero_shot_classifier(
                self.clip_model,
                self.tokenizer,
                self.colors,
                [
                    f"a photo of a {{c}} {classname}",
                    f"a photo of a {{c}}-colored {classname}",
                    f"a photo of a {{c}} object",
                ],
                self.device,
            )
        clf = self.color_classifiers[classname]
        dataloader = torch.utils.data.DataLoader(
            self._ImageCrops(image, bboxes, self.transform),
            batch_size=16,
            num_workers=4,
        )
        with torch.no_grad():
            pred, _ = self.zsc.run_classification(self.clip_model, clf, dataloader, self.device)
        return [self.colors[index.item()] for index in pred.argmax(1)]

    @staticmethod
    def _compute_iou(box_a, box_b) -> float:
        area_fn = lambda box: max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)
        i_area = area_fn(
            [
                max(box_a[0], box_b[0]),
                max(box_a[1], box_b[1]),
                min(box_a[2], box_b[2]),
                min(box_a[3], box_b[3]),
            ]
        )
        u_area = area_fn(box_a) + area_fn(box_b) - i_area
        return i_area / u_area if u_area else 0.0

    def _relative_position(self, obj_a, obj_b) -> set:
        boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
        center_a, center_b = boxes.mean(axis=-2)
        dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
        offset = center_a - center_b
        revised_offset = (
            np.maximum(np.abs(offset) - self.position_threshold * (dim_a + dim_b), 0) * np.sign(offset)
        )
        if np.all(np.abs(revised_offset) < 1e-3):
            return set()

        dx, dy = revised_offset / np.linalg.norm(offset)
        relations = set()
        if dx < -0.5:
            relations.add("left of")
        if dx > 0.5:
            relations.add("right of")
        if dy < -0.5:
            relations.add("above")
        if dy > 0.5:
            relations.add("below")
        return relations

    def _evaluate(self, image: Image.Image, objects: Dict[str, list], metadata: Dict) -> Tuple[bool, str]:
        correct = True
        reason = []
        matched_groups = []
        for req in metadata.get("include", []):
            classname = req["class"]
            matched = True
            found_objects = objects.get(classname, [])[: req["count"]]
            if len(found_objects) < req["count"]:
                correct = matched = False
                reason.append(f"expected {classname}>={req['count']}, found {len(found_objects)}")
            else:
                if "color" in req:
                    colors = self._color_classification(image, found_objects, classname)
                    if colors.count(req["color"]) < req["count"]:
                        correct = matched = False
                        reason.append(
                            f"expected {req['color']} {classname}>={req['count']}, found "
                            f"{colors.count(req['color'])} {req['color']}"
                        )
                if "position" in req and matched:
                    expected_rel, target_group = req["position"]
                    if matched_groups[target_group] is None:
                        correct = matched = False
                        reason.append(f"no target for {classname} to be {expected_rel}")
                    else:
                        for obj in found_objects:
                            for target_obj in matched_groups[target_group]:
                                true_rels = self._relative_position(obj, target_obj)
                                if expected_rel not in true_rels:
                                    correct = matched = False
                                    reason.append(
                                        f"expected {classname} {expected_rel} target, found {' and '.join(true_rels)} target"
                                    )
                                    break
                            if not matched:
                                break
            matched_groups.append(found_objects if matched else None)
        for req in metadata.get("exclude", []):
            classname = req["class"]
            if len(objects.get(classname, [])) >= req["count"]:
                correct = False
                reason.append(f"expected {classname}<{req['count']}, found {len(objects[classname])}")
        return correct, "\n".join(reason)

    def _evaluate_reward(self, image: Image.Image, objects: Dict[str, list], metadata: Dict) -> Tuple[bool, float, str]:
        correct = True
        reason = []
        rewards = []
        matched_groups = []
        for req in metadata.get("include", []):
            classname = req["class"]
            matched = True
            found_objects = objects.get(classname, [])
            rewards.append(1 - abs(req["count"] - len(found_objects)) / req["count"])
            if len(found_objects) != req["count"]:
                correct = matched = False
                reason.append(f"expected {classname}=={req['count']}, found {len(found_objects)}")
                if "color" in req or "position" in req:
                    rewards.append(0.0)
            else:
                if "color" in req:
                    colors = self._color_classification(image, found_objects, classname)
                    rewards.append(1 - abs(req["count"] - colors.count(req["color"])) / req["count"])
                    if colors.count(req["color"]) != req["count"]:
                        correct = matched = False
                        reason.append(
                            f"expected {req['color']} {classname}>={req['count']}, found "
                            f"{colors.count(req['color'])} {req['color']}"
                        )
                if "position" in req and matched:
                    expected_rel, target_group = req["position"]
                    if matched_groups[target_group] is None:
                        correct = matched = False
                        reason.append(f"no target for {classname} to be {expected_rel}")
                        rewards.append(0.0)
                    else:
                        relation_matched = True
                        for obj in found_objects:
                            for target_obj in matched_groups[target_group]:
                                true_rels = self._relative_position(obj, target_obj)
                                if expected_rel not in true_rels:
                                    correct = matched = relation_matched = False
                                    reason.append(
                                        f"expected {classname} {expected_rel} target, found {' and '.join(true_rels)} target"
                                    )
                                    break
                            if not relation_matched:
                                break
                        rewards.append(1.0 if relation_matched else 0.0)
            matched_groups.append(found_objects if matched else None)

        reward = sum(rewards) / len(rewards) if rewards else 0.0
        return correct, reward, "\n".join(reason)

    def _extract_detector_outputs(self, result):
        if isinstance(result, tuple):
            bbox = result[0]
            segm = result[1] if len(result) > 1 else None
            return bbox, segm

        pred_instances = getattr(result, "pred_instances", None)
        if pred_instances is None:
            return result, None

        bboxes = pred_instances.bboxes.detach().cpu().numpy()
        scores = pred_instances.scores.detach().cpu().numpy().reshape(-1, 1)
        labels = pred_instances.labels.detach().cpu().numpy().astype(np.int64)
        bbox_by_class = [np.zeros((0, 5), dtype=np.float32) for _ in self.classnames]

        masks = getattr(pred_instances, "masks", None)
        if masks is not None:
            masks = masks.detach().cpu().numpy()
        segm_by_class = [[] for _ in self.classnames] if masks is not None else None

        for det_index, class_index in enumerate(labels.tolist()):
            box_with_score = np.concatenate([bboxes[det_index], scores[det_index]], axis=0).astype(np.float32)
            bbox_by_class[class_index] = np.vstack([bbox_by_class[class_index], box_with_score])
            if segm_by_class is not None:
                segm_by_class[class_index].append(masks[det_index])

        return bbox_by_class, segm_by_class

    def _evaluate_image(self, image_pils: Sequence[Image.Image], metadatas: Sequence[Dict], only_strict: bool):
        results = self.inference_detector(self.object_detector, [np.array(image_pil) for image_pil in image_pils])
        ret = []
        for result, image_pil, metadata in zip(results, image_pils, metadatas):
            bbox, segm = self._extract_detector_outputs(result)
            image = ImageOps.exif_transpose(image_pil)
            detected = {}
            confidence_threshold = self.threshold if metadata["tag"] != "counting" else self.counting_threshold
            for index, classname in enumerate(self.classnames):
                ordering = np.argsort(bbox[index][:, 4])[::-1]
                ordering = ordering[bbox[index][ordering, 4] > confidence_threshold]
                ordering = ordering[: self.max_objects].tolist()
                detected[classname] = []
                while ordering:
                    max_obj = ordering.pop(0)
                    detected[classname].append((bbox[index][max_obj], None if segm is None else segm[index][max_obj]))
                    ordering = [
                        obj for obj in ordering
                        if self.nms_threshold == 1 or self._compute_iou(bbox[index][max_obj], bbox[index][obj]) < self.nms_threshold
                    ]
                if not detected[classname]:
                    del detected[classname]

            is_strict_correct, score, reason = self._evaluate_reward(image, detected, metadata)
            if only_strict:
                is_correct = False
            else:
                is_correct, _ = self._evaluate(image, detected, metadata)
            ret.append(
                {
                    "tag": metadata["tag"],
                    "prompt": metadata["prompt"],
                    "correct": is_correct,
                    "strict_correct": is_strict_correct,
                    "score": score,
                    "reason": reason,
                    "metadata": json.dumps(metadata),
                    "details": json.dumps({key: [box.tolist() for box, _ in value] for key, value in detected.items()}),
                }
            )
        return ret

    @torch.no_grad()
    def score(self, images: Sequence[Image.Image], metadatas: Sequence[Dict], only_strict: bool = True):
        required_keys = ["single_object", "two_object", "counting", "colors", "position", "color_attr"]
        scores = []
        strict_rewards = []
        grouped_strict_rewards = defaultdict(list)
        rewards = []
        grouped_rewards = defaultdict(list)
        details = []

        results = self._evaluate_image(images, metadatas, only_strict=only_strict)
        for result in results:
            score = min(1.0, max(0.0, float(result["score"])))
            result["score"] = score
            strict_rewards.append(1.0 if result["strict_correct"] else 0.0)
            scores.append(score)
            rewards.append(1.0 if result["correct"] else 0.0)
            details.append(result)
            tag = result["tag"]
            for key in required_keys:
                if key != tag:
                    grouped_strict_rewards[key].append(-10.0)
                    grouped_rewards[key].append(-10.0)
                else:
                    grouped_strict_rewards[tag].append(1.0 if result["strict_correct"] else 0.0)
                    grouped_rewards[tag].append(1.0 if result["correct"] else 0.0)
        return scores, rewards, strict_rewards, dict(grouped_rewards), dict(grouped_strict_rewards), details

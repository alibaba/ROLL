"""
reference: https://github.com/MiniMax-AI/One-RL-to-See-Them-All/blob/main/reward_server/verify.py
"""

import copy
import datetime
import itertools
import json
import logging
import os
import re
import time
from collections import defaultdict
from typing import Optional, Union, List, Tuple, Dict

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from urllib.request import urlretrieve

try:
    from pycocotools import mask as maskUtils
except:
    maskUtils = None

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import default_tokenizer_provider
from roll.utils.logging import get_logger


logger = get_logger()


def _isArrayLike(obj):
    return hasattr(obj, "__iter__") and hasattr(obj, "__len__")


class COCO:
    def __init__(self, annotation_file: Union[Dict, str] = None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            # logger.info('loading annotations into memory...')
            tic = time.time()
            if type(annotation_file) == dict:
                dataset = annotation_file
            else:
                dataset = json.load(open(annotation_file, "r"))
            assert type(dataset) == dict, "annotation file format {} not supported".format(type(dataset))
            # logger.info('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        # logger.info('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        if "annotations" in self.dataset:
            for ann in self.dataset["annotations"]:
                imgToAnns[ann["image_id"]].append(ann)
                anns[ann["id"]] = ann

        if "images" in self.dataset:
            for img in self.dataset["images"]:
                imgs[img["id"]] = img

        if "categories" in self.dataset:
            for cat in self.dataset["categories"]:
                cats[cat["id"]] = cat

        if "annotations" in self.dataset and "categories" in self.dataset:
            for ann in self.dataset["annotations"]:
                catToImgs[ann["category_id"]].append(ann["image_id"])

        # logger.info('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset["info"].items():
            logger.info("{}: {}".format(key, value))

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset["annotations"]
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset["annotations"]
            anns = anns if len(catIds) == 0 else [ann for ann in anns if ann["category_id"] in catIds]
            anns = (
                anns
                if len(areaRng) == 0
                else [ann for ann in anns if ann["area"] > areaRng[0] and ann["area"] < areaRng[1]]
            )
        if not iscrowd == None:
            ids = [ann["id"] for ann in anns if ann["iscrowd"] == iscrowd]
        else:
            ids = [ann["id"] for ann in anns]
        return ids

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset["categories"]
        else:
            cats = self.dataset["categories"]
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat["name"] in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat["supercategory"] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat["id"] in catIds]
        ids = [cat["id"] for cat in cats]
        return ids

    def getImgIds(self, imgIds=[], catIds=[]):
        """
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if _isArrayLike(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _isArrayLike(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]

    def showAnns(self, anns, draw_bbox=False):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        if "segmentation" in anns[0] or "keypoints" in anns[0]:
            datasetType = "instances"
        elif "caption" in anns[0]:
            datasetType = "captions"
        else:
            raise Exception("datasetType not supported")
        if datasetType == "instances":
            ax = plt.gca()
            ax.set_autoscale_on(False)
            polygons = []
            color = []
            for ann in anns:
                c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
                if "segmentation" in ann:
                    if type(ann["segmentation"]) == list:
                        # polygon
                        for seg in ann["segmentation"]:
                            poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                            polygons.append(Polygon(poly))
                            color.append(c)
                    else:
                        # mask
                        t = self.imgs[ann["image_id"]]
                        if type(ann["segmentation"]["counts"]) == list:
                            rle = maskUtils.frPyObjects([ann["segmentation"]], t["height"], t["width"])
                        else:
                            rle = [ann["segmentation"]]
                        m = maskUtils.decode(rle)
                        img = np.ones((m.shape[0], m.shape[1], 3))
                        if ann["iscrowd"] == 1:
                            color_mask = np.array([2.0, 166.0, 101.0]) / 255
                        if ann["iscrowd"] == 0:
                            color_mask = np.random.random((1, 3)).tolist()[0]
                        for i in range(3):
                            img[:, :, i] = color_mask[i]
                        ax.imshow(np.dstack((img, m * 0.5)))
                if "keypoints" in ann and type(ann["keypoints"]) == list:
                    # turn skeleton into zero-based index
                    sks = np.array(self.loadCats(ann["category_id"])[0]["skeleton"]) - 1
                    kp = np.array(ann["keypoints"])
                    x = kp[0::3]
                    y = kp[1::3]
                    v = kp[2::3]
                    for sk in sks:
                        if np.all(v[sk] > 0):
                            plt.plot(x[sk], y[sk], linewidth=3, color=c)
                    plt.plot(
                        x[v > 0],
                        y[v > 0],
                        "o",
                        markersize=8,
                        markerfacecolor=c,
                        markeredgecolor="k",
                        markeredgewidth=2,
                    )
                    plt.plot(
                        x[v > 1], y[v > 1], "o", markersize=8, markerfacecolor=c, markeredgecolor=c, markeredgewidth=2
                    )

                if draw_bbox:
                    [bbox_x, bbox_y, bbox_w, bbox_h] = ann["bbox"]
                    poly = [
                        [bbox_x, bbox_y],
                        [bbox_x, bbox_y + bbox_h],
                        [bbox_x + bbox_w, bbox_y + bbox_h],
                        [bbox_x + bbox_w, bbox_y],
                    ]
                    np_poly = np.array(poly).reshape((4, 2))
                    polygons.append(Polygon(np_poly))
                    color.append(c)

            p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
            ax.add_collection(p)
            p = PatchCollection(polygons, facecolor="none", edgecolors=color, linewidths=2)
            ax.add_collection(p)
        elif datasetType == "captions":
            for ann in anns:
                logger.info(ann["caption"])

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = COCO()
        res.dataset["images"] = [img for img in self.dataset["images"]]

        # logger.info('Loading and preparing results...')
        tic = time.time()
        if type(resFile) == str:
            anns = json.load(open(resFile))
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, "results in not an array of objects"
        annsImgIds = [ann["image_id"] for ann in anns]
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), (
            "Results do not correspond to current coco set"
        )
        if "caption" in anns[0]:
            imgIds = set([img["id"] for img in res.dataset["images"]]) & set([ann["image_id"] for ann in anns])
            res.dataset["images"] = [img for img in res.dataset["images"] if img["id"] in imgIds]
            for id, ann in enumerate(anns):
                ann["id"] = id + 1
        elif "bbox" in anns[0] and not anns[0]["bbox"] == []:
            res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
            for id, ann in enumerate(anns):
                bb = ann["bbox"]
                x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                if not "segmentation" in ann:
                    ann["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann["area"] = bb[2] * bb[3]
                ann["id"] = id + 1
                ann["iscrowd"] = 0
        elif "segmentation" in anns[0]:
            res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
            for id, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann["area"] = maskUtils.area(ann["segmentation"])
                if not "bbox" in ann:
                    ann["bbox"] = maskUtils.toBbox(ann["segmentation"])
                ann["id"] = id + 1
                ann["iscrowd"] = 0
        elif "keypoints" in anns[0]:
            res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
            for id, ann in enumerate(anns):
                s = ann["keypoints"]
                x = s[0::3]
                y = s[1::3]
                x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                ann["area"] = (x1 - x0) * (y1 - y0)
                ann["id"] = id + 1
                ann["bbox"] = [x0, y0, x1 - x0, y1 - y0]
        # logger.info('DONE (t={:0.2f}s)'.format(time.time()- tic))

        res.dataset["annotations"] = anns
        res.createIndex()
        return res

    def download(self, tarDir=None, imgIds=[]):
        """
        Download COCO images from mscoco.org server.
        :param tarDir (str): COCO results directory name
               imgIds (list): images to be downloaded
        :return:
        """
        if tarDir is None:
            logger.info("Please specify target directory")
            return -1
        if len(imgIds) == 0:
            imgs = self.imgs.values()
        else:
            imgs = self.loadImgs(imgIds)
        N = len(imgs)
        if not os.path.exists(tarDir):
            os.makedirs(tarDir)
        for i, img in enumerate(imgs):
            tic = time.time()
            fname = os.path.join(tarDir, img["file_name"])
            if not os.path.exists(fname):
                urlretrieve(img["coco_url"], fname)
            logger.info("downloaded {}/{} images (t={:0.1f}s)".format(i, N, time.time() - tic))

    def loadNumpyAnnotations(self, data):
        """
        Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
        :param  data (numpy.ndarray)
        :return: annotations (python nested list)
        """
        logger.info("Converting ndarray to lists...")
        assert type(data) == np.ndarray
        logger.info(data.shape)
        assert data.shape[1] == 7
        N = data.shape[0]
        ann = []
        for i in range(N):
            if i % 1000000 == 0:
                logger.info("{}/{}".format(i, N))
            ann += [
                {
                    "image_id": int(data[i, 0]),
                    "bbox": [data[i, 1], data[i, 2], data[i, 3], data[i, 4]],
                    "score": data[i, 5],
                    "category_id": int(data[i, 6]),
                }
            ]
        return ann

    def annToRLE(self, ann):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        t = self.imgs[ann["image_id"]]
        h, w = t["height"], t["width"]
        segm = ann["segmentation"]
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        elif type(segm["counts"]) == list:
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = ann["segmentation"]
        return rle

    def annToMask(self, ann):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann)
        m = maskUtils.decode(rle)
        return m


class COCOeval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, cocoGt=None, cocoDt=None, iouType="segm"):
        """
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        """
        if not iouType:
            logger.info("iouType not specified. use default iouType segm")
        self.cocoGt = cocoGt  # ground truth COCO API
        self.cocoDt = cocoDt  # detections COCO API
        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results [KxAxI] elements
        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params = Params(iouType=iouType)  # parameters
        self._paramsEval = {}  # parameters for evaluation
        self.stats = []  # result summarization
        self.ious = {}  # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

    def _prepare(self):
        """
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        """

        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann["segmentation"] = rle

        p = self.params
        if p.useCats:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == "segm":
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt["ignore"] = gt["ignore"] if "ignore" in gt else 0
            gt["ignore"] = "iscrowd" in gt and gt["iscrowd"]
            if p.iouType == "keypoints":
                gt["ignore"] = (gt["num_keypoints"] == 0) or gt["ignore"]
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)
        for dt in dts:
            self._dts[dt["image_id"], dt["category_id"]].append(dt)
        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results
        self.eval = {}  # accumulated evaluation results

    def evaluate(self):
        """
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        """
        tic = time.time()
        # ('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = "segm" if p.useSegm == 1 else "bbox"
            logger.info("useSegm (deprecated) is not None. Running {} evaluation".format(p.iouType))

        # logger.info('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == "segm" or p.iouType == "bbox":
            computeIoU = self.computeIoU
        elif p.iouType == "keypoints":
            computeIoU = self.computeOks
        self.ious = {(imgId, catId): computeIoU(imgId, catId) for imgId in p.imgIds for catId in catIds}
        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [
            evaluateImg(imgId, catId, areaRng, maxDet)
            for catId in catIds
            for areaRng in p.areaRng
            for imgId in p.imgIds
        ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        # logger.info('DONE (t={:0.2f}s).'.format(toc-tic))

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0 : p.maxDets[-1]]

        if p.iouType == "segm":
            g = [g["segmentation"] for g in gt]
            d = [d["segmentation"] for d in dt]
        elif p.iouType == "bbox":
            g = [g["bbox"] for g in gt]
            d = [d["bbox"] for d in dt]
        else:
            raise Exception("unknown iouType for iou computation")

        # compute iou between each dt and gt region
        iscrowd = [int(o["iscrowd"]) for o in gt]
        ious = maskUtils.iou(d, g, iscrowd)
        return ious

    def computeOks(self, imgId, catId):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d["score"] for d in dts], kind="mergesort")
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0 : p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = p.kpt_oks_sigmas
        vars = (sigmas * 2) ** 2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt["keypoints"])
            xg = g[0::3]
            yg = g[1::3]
            vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt["bbox"]
            x0 = bb[0] - bb[2]
            x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]
            y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt["keypoints"])
                xd = d[0::3]
                yd = d[1::3]
                if k1 > 0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0 - xd), axis=0) + np.max((z, xd - x1), axis=0)
                    dy = np.max((z, y0 - yd), axis=0) + np.max((z, yd - y1), axis=0)
                e = (dx**2 + dy**2) / vars / (gt["area"] + np.spacing(1)) / 2
                if k1 > 0:
                    e = e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        """
        perform evaluation for single category and image
        :return: dict (single image results)
        """
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g["ignore"] or (g["area"] < aRng[0] or g["area"] > aRng[1]):
                g["_ignore"] = 1
            else:
                g["_ignore"] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o["iscrowd"]) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g["_ignore"] for g in gt])
        dtIg = np.zeros((T, D))
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]["id"]
                    gtm[tind, m] = d["id"]
        # set unmatched detections outside of area range to ignore
        a = np.array([d["area"] < aRng[0] or d["area"] > aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
        # store results for given image and category
        return {
            "image_id": imgId,
            "category_id": catId,
            "aRng": aRng,
            "maxDet": maxDet,
            "dtIds": [d["id"] for d in dt],
            "gtIds": [g["id"] for g in gt],
            "dtMatches": dtm,
            "gtMatches": gtm,
            "dtScores": [d["score"] for d in dt],
            "gtIgnore": gtIg,
            "dtIgnore": dtIg,
        }

    def accumulate(self, p=None):
        """
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        """
        # logger.info('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            logger.info("Please run evaluate() first")
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)
        precision = -np.ones((T, R, K, A, M))  # -1 for the precision of absent categories
        recall = -np.ones((T, K, A, M))
        scores = -np.ones((T, R, K, A, M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e["dtScores"][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind="mergesort")
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate([e["dtMatches"][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    dtIg = np.concatenate([e["dtIgnore"][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    gtIg = np.concatenate([e["gtIgnore"] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist()
                        q = q.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side="left")
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t, :, k, a, m] = np.array(q)
                        scores[t, :, k, a, m] = np.array(ss)
        self.eval = {
            "params": p,
            "counts": [T, R, K, A, M],
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,
            "recall": recall,
            "scores": scores,
        }
        toc = time.time()
        # logger.info('DONE (t={:0.2f}s).'.format( toc-tic))

    def summarize(self):
        """
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        """

        def _summarize(ap=1, iouThr=None, areaRng="all", maxDets=100):
            p = self.params
            iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
            titleStr = "Average Precision" if ap == 1 else "Average Recall"
            typeStr = "(AP)" if ap == 1 else "(AR)"
            iouStr = (
                "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1]) if iouThr is None else "{:0.2f}".format(iouThr)
            )

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["precision"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval["recall"]
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            # logger.info(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=0.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=0.75, maxDets=self.params.maxDets[2])
            """
            ignore the following
            # stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            # stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            # stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            # stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            # stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            # stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            # stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            # stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            # stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            """
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=0.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=0.75)
            stats[3] = _summarize(1, maxDets=20, areaRng="medium")
            stats[4] = _summarize(1, maxDets=20, areaRng="large")
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=0.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=0.75)
            stats[8] = _summarize(0, maxDets=20, areaRng="medium")
            stats[9] = _summarize(0, maxDets=20, areaRng="large")
            return stats

        if not self.eval:
            raise Exception("Please run accumulate() first")
        iouType = self.params.iouType
        if iouType == "segm" or iouType == "bbox":
            summarize = _summarizeDets
        elif iouType == "keypoints":
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self):
        self.summarize()


class Params:
    """
    Params for coco evaluation api
    """

    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
        self.recThrs = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        """
        不考虑area分层级
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        """
        self.areaRng = [[0**2, 1e5**2]]
        self.areaRngLbl = ["all"]

        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
        self.recThrs = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0**2, 1e5**2], [32**2, 96**2], [96**2, 1e5**2]]
        self.areaRngLbl = ["all", "medium", "large"]
        self.useCats = 1
        self.kpt_oks_sigmas = (
            np.array(
                [0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89]
            )
            / 10.0
        )

    def __init__(self, iouType="segm"):
        if iouType == "segm" or iouType == "bbox":
            self.setDetParams()
        elif iouType == "keypoints":
            self.setKpParams()
        else:
            raise Exception("iouType not supported")
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None


def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: List or tuple of 4 values [x1, y1, x2, y2] representing first box
        box2: List or tuple of 4 values [x1, y1, x2, y2] representing second box

    Returns:
        float: IoU value between the two boxes (0.0 to 1.0)
    """
    # Find coordinates of intersection
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    # Check if boxes overlap
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    # Calculate intersection area
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    # Calculate areas of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate union area
    union_area = box1_area + box2_area - inter_area

    # Return IoU
    return float(inter_area) / union_area


def greedy_match_by_iou_max_iou_first(predict_bbox, answer_bbox, iou_threshold):
    """
    Use IoU as metric to perform greedy match.
    Find the maximum IoU in predict bbox for each solution (answer) bbox.
    Check if the label is correct and add IoU to the final score if correct.
    """
    iou_matrix = np.zeros((len(predict_bbox), len(answer_bbox)))

    for i in range(len(predict_bbox)):
        for j in range(len(answer_bbox)):
            iou_matrix[i][j] = compute_iou(predict_bbox[i]["bbox_2d"], answer_bbox[j]["bbox_2d"])

    # Find the maximum IoU in predict bbox for each solution bbox globally
    matches = []  # Store matched pairs of predicted and ground truth boxes
    unmatched_pred = list(range(len(predict_bbox)))  # Unmatched predicted boxes
    unmatched_gt = list(range(len(answer_bbox)))  # Unmatched ground truth boxes

    # Greedy matching: find the best match for each predicted box
    while unmatched_pred and unmatched_gt:
        # Find the maximum IoU
        max_iou, pred_idx, gt_idx = -1, -1, -1

        for pred_idx in unmatched_pred:
            for gt_idx in unmatched_gt:
                curr_iou = iou_matrix[pred_idx][gt_idx]
                #  find the largest iou in the unmatched list
                if curr_iou > max_iou:
                    max_iou, pred_idx, gt_idx = curr_iou, pred_idx, gt_idx

        # Stop matching if the maximum IoU is below the threshold
        if max_iou < iou_threshold:
            break

        # Record matching results
        pred_label = predict_bbox[pred_idx]["label"].lower()
        gt_label = answer_bbox[gt_idx]["label"].lower()
        iou_score = max_iou if pred_label == gt_label else 0.0

        matches.append({"pred_idx": pred_idx, "gt_idx": gt_idx, "iou": iou_score})

        # Remove matched boxes from the unmatched list
        unmatched_pred.remove(pred_idx)
        unmatched_gt.remove(gt_idx)

    return matches


def greedy_match_by_iou_max_label_first(predict_bbox, answer_bbox, iou_threshold):
    """
    Use IoU as metric to perform greedy match.
    First find the matched labels for both predict and answer.
    Then use max IoU to find the best match.
    """
    matches = []
    matched_pred_idx = set()
    for gt_idx, gt_bbox in enumerate(answer_bbox):
        label = gt_bbox["label"].lower()

        # Find the potential matches and IoU scores
        potential_matches, iou_scores = [], []
        for pred_idx, pred_bbox in enumerate(predict_bbox):
            if pred_idx in matched_pred_idx:
                continue

            pred_label = pred_bbox["label"].lower()
            if pred_label == label:
                iou = compute_iou(pred_bbox["bbox_2d"], gt_bbox["bbox_2d"])
                potential_matches.append(pred_idx)
                iou_scores.append(iou)

        if len(potential_matches) == 0:
            continue

        max_iou, max_iou_idx = np.max(iou_scores), np.argmax(iou_scores)
        if max_iou >= iou_threshold:
            matches.append({"pred_idx": potential_matches[max_iou_idx], "gt_idx": gt_idx, "iou": max_iou})
            matched_pred_idx.add(potential_matches[max_iou_idx])

    return matches


def extract_bbox(bbox_str, ignore_confidence=True):
    """Extract bounding box information from a string.

    Parses a string containing bbox data in the format:
    [{'bbox_2d':[x1,y1,x2,y2],'confidence':number,'label':label_name},...]

    Returns:
        List of dictionaries containing bbox information or None if parsing fails
    """
    if not bbox_str or not bbox_str.strip():
        return None

    bbox_str = bbox_str.replace("[[", "[").replace("]]", "]")
    if not bbox_str.endswith("]"):
        bbox_str = bbox_str.rsplit("},", 1)[0] + "}]"
    bbox_str = bbox_str.replace("'", '"').replace("\\", "")
    try:
        bbox_list = json.loads(bbox_str)
        if not isinstance(bbox_list, list):
            return None

        # Validate that each item has the required fields and correct types
        filtered_bbox_list = []
        seen = set()
        for item in bbox_list:
            if (
                isinstance(item, dict)
                and "bbox_2d" in item
                and (ignore_confidence or "confidence" in item)
                and "label" in item
                and isinstance(item["bbox_2d"], list)
                and len(item["bbox_2d"]) == 4
                and all(isinstance(coord, (int, float)) for coord in item["bbox_2d"])
                and item["bbox_2d"][0] <= item["bbox_2d"][2]
                and item["bbox_2d"][1] <= item["bbox_2d"][3]
                and (ignore_confidence or isinstance(item["confidence"], (int, float)))
                and isinstance(item["label"], str)
            ):
                # Remove duplicates based on bbox position
                position_tuple = tuple(item["bbox_2d"])
                if position_tuple not in seen:
                    seen.add(position_tuple)
                    filtered_bbox_list.append(item)

        if len(filtered_bbox_list) == 0:
            return None
        return filtered_bbox_list

    except Exception:
        return None


def convert_bbox_to_coco_format(bbox):
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1], (x2 - x1) * (y2 - y1)


def extract_answer_content(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def normalize_bbox_by_real_size(pred_bboxes, input_width, input_height, normalize_size=1000.0):
    if pred_bboxes is None:
        return None

    for idx, pred_bbox in enumerate(pred_bboxes):
        try:
            x1, y1, x2, y2 = pred_bbox["bbox_2d"]

            # Calculate normalized coordinates
            x1_norm = int(x1 / input_width * normalize_size)
            y1_norm = int(y1 / input_height * normalize_size)
            x2_norm = int(x2 / input_width * normalize_size)
            y2_norm = int(y2 / input_height * normalize_size)

            pred_bbox["bbox_2d"] = [x1_norm, y1_norm, x2_norm, y2_norm]
        except (KeyError, ValueError, TypeError) as e:
            # Handle case where bbox_2d is missing or malformed
            logger.info(f"Error normalizing bbox:  {e}")
            continue

    return pred_bboxes


class DetectionVerifier:
    def __init__(
        self,
        is_training: bool,
        step: int,
        total_steps: int,
        image_path: Optional[List[str]] = None,
        image_grid_thw: Optional[List[Tuple[int, int, int]]] = None,
        verifier_style: str = "rule",
        det_verifier_normalized: bool = False,
        det_reward_ratio: Dict[str, float] = {},
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger if logger else get_logger()
        # for dynamic iou threshold
        self.is_training = is_training
        self.step = step
        self.total_steps = total_steps
        self.step_ratio = float(step) / float(total_steps)
        self.image_path = image_path
        self.image_grid_thw = image_grid_thw
        self.verifier_style = verifier_style
        self.det_verifier_normalized = det_verifier_normalized
        self.det_reward_ratio = det_reward_ratio

        iou_threshold = os.environ.get("DET_IOU_THRESHOLD", None)
        if iou_threshold is None:
            self.iou_threshold = "average"
        elif iou_threshold == "average":
            self.iou_threshold = "average"
        elif iou_threshold == "dynamic":
            self.iou_threshold = "dynamic"
        else:
            try:
                self.iou_threshold = float(iou_threshold)
            except ValueError:
                self.iou_threshold = "average"

        if self.iou_threshold not in [
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
            0.99,
            "average",
            "dynamic"
        ]:
            self.logger.error("DET_IOU_THRESHOLD is not set, assign 'average' to it")
            self.iou_threshold = "average"
        else:
            self.logger.info(f"DET_IOU_THRESHOLD is set to {self.iou_threshold}")

        for key in ["iou_max_label_first", "iou_max_iou_first", "iou_completeness", "map", "map50", "map75"]:
            if key not in self.det_reward_ratio:
                self.logger.error(f"Key {key} not found in det_reward_ratio, assign 0.0 to it")
                self.det_reward_ratio[key] = 0.0
            elif self.det_reward_ratio[key] < 0 or self.det_reward_ratio[key] > 1:
                self.logger.warning(f"Value for key {key} must be between 0 and 1, assign 0.0 to it")
                self.det_reward_ratio[key] = 0.0

    def verify_format(self, predict_str: str) -> float:
        try:
            # Only one answer tag is allowed
            if predict_str.count("\n<answer>\n") and predict_str.count("\n</answer>") == 1:
                # continue to match the format
                predict_extract = extract_answer_content(predict_str.strip()).lower()
                predict_bbox = extract_bbox(predict_extract)

                # extract bbox from answer
                if predict_bbox is None or len(predict_bbox) == 0:
                    return 0.0
                else:
                    return 1.0
            else:
                return 0.0
        except Exception:
            return 0.0

    def pack_for_map_score(self, predict_bbox, answer_bbox):
        """
        predict_bbox: [{'bbox_2d': [x1, y1, x2, y2], 'label': 'label_name'}, ...]
        answer_bbox: [{'bbox_2d': [x1, y1, x2, y2], 'label': 'label_name'}, ...]
        return dict with mAP and mAP50 scores

        advice: pure map should be used with length penalty
        """

        # Initialize COCO object for ground truth
        gt_json = {"annotations": [], "images": [], "categories": []}
        gt_json["images"] = [{"id": 0, "file_name": "fake_image_0.jpg"}]

        gt_json["categories"] = []

        cats2id, cat_count = {}, 0
        for idx, gt_bbox in enumerate(answer_bbox):
            if gt_bbox["label"] not in cats2id:
                cats2id[gt_bbox["label"]] = cat_count
                gt_json["categories"].append({"id": cat_count, "name": gt_bbox["label"]})
                cat_count += 1

            bbox_coco_format, bbox_area_coco_format = convert_bbox_to_coco_format(gt_bbox["bbox_2d"])
            gt_json["annotations"].append(
                {
                    "id": idx + 1,
                    "image_id": 0,
                    "category_id": cats2id[gt_bbox["label"]],
                    "bbox": bbox_coco_format,
                    "area": bbox_area_coco_format,
                    "iscrowd": 0,
                }
            )
        # Initialize COCO object for ground truth
        coco_gt = COCO(gt_json)

        dt_json = []
        for idx, pred_bbox in enumerate(predict_bbox):
            if pred_bbox["label"] not in cats2id:
                continue
            bbox_coco_format, bbox_area_coco_format = convert_bbox_to_coco_format(pred_bbox["bbox_2d"])
            dt_json.append(
                {
                    "image_id": 0,
                    "category_id": cats2id[pred_bbox["label"]],
                    "bbox": bbox_coco_format,
                    "score": 1.0,  # no confidence score in predict_bbox right now
                    "area": bbox_area_coco_format,
                }
            )
        if len(dt_json) == 0:
            return {"map": 0.0, "map50": 0.0, "map75": 0.0}

        coco_dt = coco_gt.loadRes(dt_json)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        map_score = float(coco_eval.stats[0])
        map50_score = float(coco_eval.stats[1])
        map75_score = float(coco_eval.stats[2])
        return {"map": map_score, "map50": map50_score, "map75": map75_score}

    @staticmethod
    def calculate_iou_score(predict_bbox, answer_bbox, match_strategy, iou_threshold, completeness_weight, iou_weight):
        """
        predict_bbox: [{'bbox_2d': [x1, y1, x2, y2], 'label': 'label_name'}, ...]
        answer_bbox: [{'bbox_2d': [x1, y1, x2, y2], 'label': 'label_name'}, ...]
        return dict with iou scores

        advice: pure iou should be used with length penalty
        """

        if match_strategy == "greedy_match_by_iou_max_iou_first":
            matches = greedy_match_by_iou_max_iou_first(predict_bbox, answer_bbox, iou_threshold)
        elif match_strategy == "greedy_match_by_iou_max_label_first":
            matches = greedy_match_by_iou_max_label_first(predict_bbox, answer_bbox, iou_threshold)
        else:
            raise ValueError(f"Invalid match strategy: {match_strategy}")

        # pure iou score
        mean_iou_score = sum(match["iou"] for match in matches) / len(answer_bbox) if matches else 0.0

        # miss penalty score
        miss_rate = (len(answer_bbox) - len(matches)) / len(answer_bbox)
        # Avoid division by zero when predict_bbox is empty
        false_alarm_rate = 0.0 if len(predict_bbox) == 0 else (len(predict_bbox) - len(matches)) / len(predict_bbox)
        completeness_score = 1.0 - (miss_rate + false_alarm_rate) / 2.0

        weighted_iou_score = mean_iou_score * iou_weight + completeness_score * completeness_weight

        return {
            "mean_iou_score": mean_iou_score,
            "completeness_score": completeness_score,
            "precision": 1 - false_alarm_rate,
            "recall": 1 - miss_rate,
            "weighted_iou_score": weighted_iou_score,
        }

    def pack_for_iou_score(self, predict_bbox, answer_bbox):
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
        results = {
            "greedy_match_by_iou_max_iou_first": {},
            "greedy_match_by_iou_max_label_first": {},
        }

        completeness_weight = self.det_reward_ratio["iou_completeness"]
        iou_weight = 1.0 - completeness_weight

        for strategy, strategy_results in results.items():
            for iou_threshold in iou_thresholds:
                iou_scores = self.calculate_iou_score(
                    predict_bbox, answer_bbox, strategy, iou_threshold, completeness_weight, iou_weight
                )
                strategy_results[iou_threshold] = iou_scores

        # calculate mean iou score for each strategy
        for _, strategy_results in results.items():
            strategy_results["average"] = {
                "weighted_iou_score": sum(
                    strategy_results[iou_threshold]["weighted_iou_score"] for iou_threshold in iou_thresholds
                )
                / len(iou_thresholds)
            }
        return results

    def verify_accuracy(self, predict_str: str, solution: str, return_dict=True) -> Dict[str, float]:
        if return_dict:
            result_dict = {
                "final_score": 0.0,
                "size_penalty": 0.0,
                "pure_iou_score_max_iou": 0.0,
                "pure_iou_score_max_label": 0.0,
                "pure_map_score": 0.0,
                "pure_map50_score": 0.0,
                "pure_map75_score": 0.0,
                "weighted_iou_score_max_iou": 0.0,
                "weighted_iou_score_max_label": 0.0,
                "weighted_map_score": 0.0,
                "weighted_map50_score": 0.0,
                "weighted_map75_score": 0.0,
                "report_iou_50_iou_match_pure": 0.0,
                "report_iou_50_iou_match_precision": 0.0,
                "report_iou_50_iou_match_recall": 0.0,
                "report_iou_95_iou_match_pure": 0.0,
                "report_iou_95_iou_match_precision": 0.0,
                "report_iou_95_iou_match_recall": 0.0,
                "report_iou_50_label_match_pure": 0.0,
                "report_iou_50_label_match_precision": 0.0,
                "report_iou_50_label_match_recall": 0.0,
                "report_iou_75_label_match_pure": 0.0,
                "report_iou_75_label_match_precision": 0.0,
                "report_iou_75_label_match_recall": 0.0,
                "report_iou_95_label_match_pure": 0.0,
                "report_iou_95_label_match_precision": 0.0,
                "report_iou_95_label_match_recall": 0.0,
                "report_iou_99_label_match_pure": 0.0,
                "report_iou_99_label_match_precision": 0.0,
                "report_iou_99_label_match_recall": 0.0,
            }

        predict_extract = extract_answer_content(predict_str.strip()).lower()
        answer = extract_answer_content(solution.strip()).lower()

        # for both predict and answer, ignore confidence
        predict_bbox = extract_bbox(predict_extract, ignore_confidence=True)
        answer_bbox = extract_bbox(answer, ignore_confidence=True)

        if answer_bbox is None:
            self.logger.warning(f"Check GT! No bbox found in ground truth: {solution}")
            return result_dict if return_dict else 0.0

        if predict_bbox is None:
            format_score = self.verify_format(predict_str)
            if format_score == 0.0:
                return result_dict if return_dict else 0.0
            else:
                self.logger.warning(
                    f"Potential format error! Format score: {format_score}, but no bbox found in predict_str: {predict_str}"
                )
                return result_dict if return_dict else 0.0

        if self.det_verifier_normalized:
            if self.image_grid_thw is not None and len(self.image_grid_thw) > 0:
                predict_bbox = normalize_bbox_by_real_size(
                    pred_bboxes=predict_bbox,
                    input_width=self.image_grid_thw[0][2],
                    input_height=self.image_grid_thw[0][1],
                    normalize_size=1000.0,
                )
            else:
                self.logger.warning("No image grid thw found in verifier_parm")

        # Handle empty predict_bbox after normalization
        if len(predict_bbox) == 0:
            return result_dict if return_dict else 0.0

        # size_penalty ranges from (0, 1.0], 1.0 when lengths match, approaches 0 as difference increases
        size_penalty_ratio = 0.6
        size_penalty = size_penalty_ratio ** (abs(len(predict_bbox) - len(answer_bbox)))

        # ==================iou score=================
        # {'mean_iou_score': mean_iou_score, 'completeness_score': completeness_score, 'precision': 1- false_alarm_rate, 'recall': 1- miss_rate, 'weighted_iou_score': weighted_iou_score}}
        gathered_iou_scores = self.pack_for_iou_score(predict_bbox, answer_bbox)

        report_iou_50_iou_match = gathered_iou_scores["greedy_match_by_iou_max_iou_first"][0.5]
        report_iou_50_iou_match_pure, report_iou_50_iou_match_precision, report_iou_50_iou_match_recall = (
            report_iou_50_iou_match["mean_iou_score"],
            report_iou_50_iou_match["precision"],
            report_iou_50_iou_match["recall"],
        )
        report_iou_95_iou_match = gathered_iou_scores["greedy_match_by_iou_max_iou_first"][0.95]
        report_iou_95_iou_match_pure, report_iou_95_iou_match_precision, report_iou_95_iou_match_recall = (
            report_iou_95_iou_match["mean_iou_score"],
            report_iou_95_iou_match["precision"],
            report_iou_95_iou_match["recall"],
        )

        report_iou_50_label_match = gathered_iou_scores["greedy_match_by_iou_max_label_first"][0.5]
        report_iou_50_label_match_pure, report_iou_50_label_match_precision, report_iou_50_label_match_recall = (
            report_iou_50_label_match["mean_iou_score"],
            report_iou_50_label_match["precision"],
            report_iou_50_label_match["recall"],
        )
        report_iou_75_label_match = gathered_iou_scores["greedy_match_by_iou_max_label_first"][0.75]
        report_iou_75_label_match_pure, report_iou_75_label_match_precision, report_iou_75_label_match_recall = (
            report_iou_75_label_match["mean_iou_score"],
            report_iou_75_label_match["precision"],
            report_iou_75_label_match["recall"],
        )
        report_iou_95_label_match = gathered_iou_scores["greedy_match_by_iou_max_label_first"][0.95]
        report_iou_95_label_match_pure, report_iou_95_label_match_precision, report_iou_95_label_match_recall = (
            report_iou_95_label_match["mean_iou_score"],
            report_iou_95_label_match["precision"],
            report_iou_95_label_match["recall"],
        )
        report_iou_99_label_match = gathered_iou_scores["greedy_match_by_iou_max_label_first"][0.99]
        report_iou_99_label_match_pure, report_iou_99_label_match_precision, report_iou_99_label_match_recall = (
            report_iou_99_label_match["mean_iou_score"],
            report_iou_99_label_match["precision"],
            report_iou_99_label_match["recall"],
        )

        # calculate the report score
        if self.iou_threshold == "dynamic":
            if self.step_ratio <= 0.1:
                select_iou = 0.85
            elif self.step_ratio <= 0.25:
                select_iou = 0.95
            else:
                select_iou = 0.99
        else:
            select_iou = self.iou_threshold

        pure_iou_score_max_iou = gathered_iou_scores["greedy_match_by_iou_max_iou_first"][select_iou][
            "weighted_iou_score"
        ]
        pure_iou_score_max_label = gathered_iou_scores["greedy_match_by_iou_max_label_first"][select_iou][
            "weighted_iou_score"
        ]
        weighted_iou_score_max_iou = pure_iou_score_max_iou * size_penalty
        weighted_iou_score_max_label = pure_iou_score_max_label * size_penalty

        # ==================map score=================
        # {'map': map_score, 'map50': map50_score, 'map75': map75_score}
        map_score = self.pack_for_map_score(predict_bbox, answer_bbox)

        pure_map_score = map_score["map"]
        pure_map50_score = map_score["map50"]
        pure_map75_score = map_score["map75"]

        weighted_map_score = pure_map_score * size_penalty
        weighted_map50_score = pure_map50_score * size_penalty
        weighted_map75_score = pure_map75_score * size_penalty

        # Check for zero normalization factor to avoid division by zero
        normalization_factor = (
            float(self.det_reward_ratio["iou_max_iou_first"])
            + float(self.det_reward_ratio["iou_max_label_first"])
            + float(self.det_reward_ratio["map"])
            + float(self.det_reward_ratio["map50"])
            + float(self.det_reward_ratio["map75"])
        )

        self.logger.info("det_reward_ratio: %s", self.det_reward_ratio)
        # map75 is not used in the final score
        if normalization_factor > 0:
            final_det_score = (
                weighted_iou_score_max_iou * float(self.det_reward_ratio["iou_max_iou_first"])
                + weighted_iou_score_max_label * float(self.det_reward_ratio["iou_max_label_first"])
                + weighted_map_score * float(self.det_reward_ratio["map"])
                + weighted_map50_score * float(self.det_reward_ratio["map50"])
                + weighted_map75_score * float(self.det_reward_ratio["map75"])
            )
            final_det_score /= normalization_factor
        else:
            self.logger.error("Normalization factor is zero, set final score to 0.0")
            final_det_score = 0.0

        return (
            {
                "final_score": final_det_score,
                "size_penalty": size_penalty,
                "pure_iou_score_max_iou": pure_iou_score_max_iou,
                "pure_iou_score_max_label": pure_iou_score_max_label,
                "pure_map_score": pure_map_score,
                "pure_map50_score": pure_map50_score,
                "pure_map75_score": pure_map75_score,
                "weighted_iou_score_max_iou": weighted_iou_score_max_iou,
                "weighted_iou_score_max_label": weighted_iou_score_max_label,
                "weighted_map_score": weighted_map_score,
                "weighted_map50_score": weighted_map50_score,
                "weighted_map75_score": weighted_map75_score,
                "report_iou_50_iou_match_pure": report_iou_50_iou_match_pure,
                "report_iou_50_iou_match_precision": report_iou_50_iou_match_precision,
                "report_iou_50_iou_match_recall": report_iou_50_iou_match_recall,
                "report_iou_95_iou_match_pure": report_iou_95_iou_match_pure,
                "report_iou_95_iou_match_precision": report_iou_95_iou_match_precision,
                "report_iou_95_iou_match_recall": report_iou_95_iou_match_recall,
                "report_iou_50_label_match_pure": report_iou_50_label_match_pure,
                "report_iou_50_label_match_precision": report_iou_50_label_match_precision,
                "report_iou_50_label_match_recall": report_iou_50_label_match_recall,
                "report_iou_75_label_match_pure": report_iou_75_label_match_pure,
                "report_iou_75_label_match_precision": report_iou_75_label_match_precision,
                "report_iou_75_label_match_recall": report_iou_75_label_match_recall,
                "report_iou_95_label_match_pure": report_iou_95_label_match_pure,
                "report_iou_95_label_match_precision": report_iou_95_label_match_precision,
                "report_iou_95_label_match_recall": report_iou_95_label_match_recall,
                "report_iou_99_label_match_pure": report_iou_99_label_match_pure,
                "report_iou_99_label_match_precision": report_iou_99_label_match_precision,
                "report_iou_99_label_match_recall": report_iou_99_label_match_recall,
            }
            if return_dict
            else final_det_score
        )


class DetectionRewardWorker(Worker):

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.worker_config = worker_config
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None

        self.patch_size = 14  # hard-code to qwen2.5-vl temporarily

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        pass

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data: DataProto):
        response_text_list = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)
        reward_model_list = data.non_tensor_batch["reward_model"]
        multi_modal_inputs = data.non_tensor_batch["multi_modal_inputs"]
        scores = []
        for response, reward_model, multi_modal_feature in zip(
            response_text_list, reward_model_list, multi_modal_inputs
        ):
            ground_truth = reward_model["ground_truth"]
            format_ratio = reward_model.get("format_ratio", 0.1)
            accuracy_ratio = reward_model.get("accuracy_ratio", 1.0)
            verifier_parm = reward_model["verifier_parm"]
            # prepare reward verifier parm
            verifier_parm["is_training"] = data.meta_info.get("is_training", True)
            verifier_parm["step"] = data.meta_info.get("global_step", 0)
            verifier_parm["total_steps"] = data.meta_info.get("max_steps", 0)
            image_grid_thw = None
            if "image_grid_thw" in multi_modal_feature:
                image_grid_thw = multi_modal_feature["image_grid_thw"].numpy()
                image_grid_thw = [
                    (int(t), int(h * self.patch_size), int(w * self.patch_size)) for t, h, w in image_grid_thw
                ]
            verifier_parm["image_grid_thw"] = image_grid_thw
            verifier = DetectionVerifier(**verifier_parm)
            # Initialize default result
            result = {
                "rewards": {
                    "format_reward": 0.0,
                    "accuracy_reward": 0.0,
                    "reflection_reward": 0.0,
                    "final_reward": 0.0,
                }
            }
            format_score = verifier.verify_format(response)
            accuracy_score_gathered = verifier.verify_accuracy(response, ground_truth)
            if isinstance(accuracy_score_gathered, dict):
                accuracy_score = accuracy_score_gathered['final_score']

                # to log the score of each metric
                for socre_key, socre_value in accuracy_score_gathered.items():
                    if socre_key != 'final_score':
                        result['rewards'][f'{socre_key}_reward'] = socre_value
            else:
                accuracy_score = accuracy_score_gathered

            result["rewards"]["format_reward"] = float(format_score)
            result["rewards"]["accuracy_reward"] = float(accuracy_score)

            normalzied_score = accuracy_ratio + format_ratio
            # final reward
            result["rewards"]["final_reward"] = accuracy_score * (accuracy_ratio / normalzied_score) + format_score * (
                format_ratio / normalzied_score
            )
            scores.append(result["rewards"]["final_reward"])

        token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float16)
        response_level_rewards = torch.tensor(scores, dtype=torch.float16)
        scores = torch.tensor(scores, dtype=torch.float16)

        output = DataProto.from_dict(
            tensors={
                "token_level_rewards": token_level_rewards,
                "response_level_rewards": response_level_rewards,
                "scores": scores,
            }
        )
        return output

"""A simple, flexible implementation of a face analysis tool.

Inspired by https://github.com/deepinsight/insightface
"""
import torch
import numpy as np
import onnx
import cv2
import math
from onnx2torch import convert
from torchvision.transforms.functional import to_tensor, resize
import torch.nn.functional as F
from skimage import transform as trans
import time
import os
import torchvision.ops as ops
arcface_dst = torch.tensor(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]]).float()

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return torch.stack(preds, axis=-1)


def face_transform(data, center, output_size, scale, rotation, device):
    def to_homogeneous(mat):
        """将 2x3 仿射矩阵转为 3x3 齐次矩阵"""
        return torch.vstack([mat, torch.tensor([0., 0., 1.])])
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio

    C, H, W = data.shape

    # 构建各个变换矩阵
    t1 = to_homogeneous(torch.tensor([
        [scale_ratio, 0, 0],
        [0, scale_ratio, 0]
    ])).float()
    t2 = to_homogeneous(torch.tensor([
        [1, 0, -cx],
        [0, 1, -cy]
    ])).float()
    cos_theta = math.cos(rot)
    sin_theta = math.sin(rot)
    t3 = to_homogeneous(torch.tensor([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0]
    ])).float()
    t4 = to_homogeneous(torch.tensor([
        [1, 0, output_size / 2],
        [0, 1, output_size / 2]
    ])).float()
    M_homogeneous = t4 @ t3 @ t2 @ t1
    M = M_homogeneous[:2, :]  # 提取前两行作为 2x3 仿射矩阵
    # 应用仿射变换
    T = torch.tensor([[2 / W, 0, -1],
              [0, 2 / H, -1],
              [0, 0, 1]])
    theta = torch.inverse(T @ M_homogeneous @ torch.inverse(T))
    theta = theta[:2, :].unsqueeze(0).to(device)
    # theta = M.unsqueeze(0)  # 添加 batch 维度 (1, 2, 3)
    grid = F.affine_grid(theta, data.unsqueeze(0).size(), align_corners=True)
    transformed = F.grid_sample(data.unsqueeze(0), grid, align_corners=True)
    cropped = transformed[0]
    cropped = cropped[:,:output_size,:output_size]
    # crop_map = torch.zeros(3, output_size, output_size)
    # crop_map[:, :cropped.shape[1],:cropped.shape[2]] = cropped
    return cropped.unsqueeze(0), M

def trans_points2d(pts, M):
    ones = torch.ones((pts.shape[0], 1), dtype=pts.dtype, device=pts.device)
    points_hom = torch.cat([pts, ones], dim=1)  # shape: (n, 3)
    points_hom = points_hom.unsqueeze(-1)  # shape: (n, 3, 1)
    transformed_hom = torch.matmul(M, points_hom)  # shape: (n, 3, 1)
    transformed = transformed_hom[:, :2, :].squeeze(-1)  # shape: (n, 2)
    return transformed

def estimate_norm(lmk, image_size=112,mode='arcface'):
    assert lmk.shape == (5, 2)
    assert image_size%112==0 or image_size%128==0
    if image_size%112==0:
        ratio = float(image_size)/112.0
        diff_x = 0
    else:
        ratio = float(image_size)/128.0
        diff_x = 8.0*ratio
    dst = arcface_dst * ratio
    dst[:,0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = torch.from_numpy(tform.params).float()
    return M

def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M_homogeneous = estimate_norm(landmark, image_size, mode)
    C, H, W = img.shape
    img = img.unsqueeze(0)
    T = torch.tensor([[2 / W, 0, -1],
              [0, 2 / H, -1],
              [0, 0, 1]])
    T_inv = torch.inverse(T)
    theta = torch.inverse(T @ M_homogeneous @ T_inv)
    theta = theta[:2, :].unsqueeze(0).to(img.device)
    grid = F.affine_grid(theta, img.size(), align_corners=True)
    transformed = F.grid_sample(img, grid, align_corners=True)
    cropped = transformed[0]
    warped = cropped[:,:image_size,:image_size]
    return warped

def invert_affine_transform(matrix):
    L = matrix[..., :2]  # Shape: (*, 2, 2)
    T = matrix[..., 2:] # Shape: (*, 2, 1)
    a, b = L[..., 0, 0], L[..., 0, 1]
    c, d = L[..., 1, 0], L[..., 1, 1]
    det = a * d - b * c
    inv_det = 1.0 / det
    inv_L = torch.stack([
        torch.stack([d * inv_det, -b * inv_det], dim=-1),
        torch.stack([-c * inv_det, a * inv_det], dim=-1)
    ], dim=-2)  
    inv_T = -torch.matmul(inv_L, T)  
    inv_matrix = torch.cat([inv_L, inv_T], dim=-1)  
    return inv_matrix
    
class Face(dict):

    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        #for k in self.__class__.__dict__.keys():
        #    if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
        #        setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                    if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(Face, self).__setattr__(name, value)
        super(Face, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def __getattr__(self, name):
        return None

    @property
    def embedding_norm(self):
        if self.embedding is None:
            return None
        return torch.norm(self.embedding)

    @property 
    def normed_embedding(self):
        if self.embedding is None:
            return None
        return self.embedding / self.embedding_norm


class SCRFD:
    def __init__(self, model_file=None, device="cuda"):
        self.model_file = model_file
        self.device = device
        self.center_cache = {}
        self.nms_thresh = 0.4
        self.det_thresh = 0.5
        model = onnx.load(self.model_file)
        self.torch_model = convert(model)
        self.torch_model.eval()
        self.torch_model.requires_grad_(False)
        self.torch_model.to(self.device)
        self.use_kps = True
        self.fmc = 3
        self._num_anchors = 2
        self._feat_stride_fpn = [8, 16, 32]
        self.input_size = (640, 640)

    def forward(self, det_img, threshold=0.5):
        input_height = det_img.shape[2]
        input_width = det_img.shape[3]
        scores_list = []
        bboxes_list = []
        kpss_list = []
        net_outs = self.torch_model(det_img.float())
        
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx].cpu()
            bbox_preds = net_outs[idx + self.fmc].cpu()
            bbox_preds = bbox_preds * stride
            if self.use_kps:
                kps_preds = net_outs[idx + self.fmc * 2].cpu() * stride
            
            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                rows = torch.arange(height)
                cols = torch.arange(width)
                grid_y, grid_x = torch.meshgrid(rows, cols, indexing='ij')
                anchor_centers = torch.stack([grid_x, grid_y], dim=-1).float()
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors>1:
                    anchor_centers = torch.stack([anchor_centers]*self._num_anchors, axis=1).reshape( (-1,2) )
                if len(self.center_cache)<100:
                    self.center_cache[key] = anchor_centers
            
            pos_inds = np.where(scores>=threshold)[0]
            # print(bbox_preds.shape)
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                #kpss = kps_preds
                kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)

        return scores_list, bboxes_list, kpss_list
    
    @torch.no_grad()
    def detect(self, image, input_size = None, max_num = 0, metric='default'):
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size

        im_ratio = float(image.shape[1]) / image.shape[2]
        model_ratio = float(input_size[1]) / input_size[0]      
        if im_ratio>model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / image.shape[1]
        resized_img = resize(image, (new_height, new_width), antialias=True)
        det_img = torch.zeros( (3, input_size[1], input_size[0]),device=self.device)
        det_img[:, :new_height, :new_width] = resized_img
        det_img = det_img.unsqueeze(0)
        scores_list, bboxes_list, kpss_list = self.forward(det_img, self.det_thresh)

        scores = torch.vstack(scores_list)
        scores_ravel = scores.flatten()
        order = torch.argsort(scores_ravel, descending=True)
        bboxes = torch.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = torch.vstack(kpss_list) / det_scale

        pre_det = torch.cat((bboxes, scores), dim=1).float()
        pre_det = pre_det[order]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]

        if self.use_kps:
            kpss = kpss[order,:,:]
            kpss = kpss[keep,:,:]
        else:
            kpss = None
        return det, kpss
    
    def nms(self, dets):
        boxes = dets[:, :4]
        scores = dets[:, 4]
        keep = ops.nms(boxes, scores, iou_threshold=self.nms_thresh)
        return keep.tolist()


class ArcFace:
    def __init__(self, model_file=None, device="cuda"):
        self.model_file = model_file
        self.device = device
        model = onnx.load(self.model_file)
        self.torch_model = convert(model)
        self.torch_model.eval()
        self.torch_model.to(self.device)
        self.torch_model.requires_grad_(False)
        self.taskname = 'recognition'
        self.input_size = (112, 112)

    def get(self, img, face, input_size=(112, 112)):
        aimg = norm_crop(img, landmark=face.kps, image_size=self.input_size[0])
        im_ratio = float(aimg.shape[1]) / aimg.shape[2]
        model_ratio = float(input_size[1]) / input_size[0]      
        if im_ratio>model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        resized_img = resize(aimg, (new_height, new_width), antialias=True)
        face.embedding = self.get_feat(resized_img.unsqueeze(0)).flatten()
        return face.embedding

    def compute_sim(self, feat1, feat2):
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        sim = torch.dot(feat1, feat2) / (torch.norm(feat1) * torch.norm(feat2))
        return sim

    def get_feat(self, imgs):
        imgs = imgs[:,[2,1,0],:,:]
        net_out = self.torch_model(imgs)
        return net_out

class Landmark:
    def __init__(self, model_file=None, device="cuda"):
        self.model_file = model_file
        self.device = device
        model = onnx.load(self.model_file)
        self.torch_model = convert(model)
        self.torch_model.eval()
        self.torch_model.to(device)
        self.torch_model.requires_grad_(False)
        self.lmk_dim = 2
        self.lmk_num = 106
        self.taskname = 'landmark_%dd_%d'%(self.lmk_dim, self.lmk_num)
        self.input_size = (192, 192)

    def get(self, img, face, input_size=(192, 192)):
        bbox = face.bbox
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        rotate = 0
        _scale = self.input_size[0]  / (max(w, h)*1.5)
        aimg, M = face_transform(img, center, self.input_size[0], _scale, rotate, img.device)
        aimg = (aimg + 1)/2 * 255. # [1, 3, 192, 192]
        aimg = aimg[:,[2,1,0],:,:]

        input_size = self.input_size if input_size is None else input_size
        im_ratio = float(aimg.shape[2]) / aimg.shape[3]
        model_ratio = float(input_size[1]) / input_size[0]      
        if im_ratio>model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / aimg.shape[2]
        resized_img = resize(aimg, (new_height, new_width), antialias=True)
        det_img = torch.zeros( (aimg.shape[0], 3, input_size[1], input_size[0]), device=self.device)
        det_img[:, :, :new_height, :new_width] = resized_img

        pred = self.torch_model(det_img)[0] #输入图像应为RGB，不能是BGR
        pred = pred.reshape((-1, 2))
        if self.lmk_num < pred.shape[0]:
            pred = pred[self.lmk_num*-1:,:]
        pred[:, 0:2] += 1
        pred[:, 0:2] *= (self.input_size[0] // 2)
        
        IM = invert_affine_transform(M).to(img.device)
        pred = trans_points2d(pred, IM)
        face[self.taskname] = pred
        return pred

class FaceAnalysis:
    def __init__(self, root="~/.insightface", device="cuda"):
        self.root = root
        self.device = device
        self.detection_root = os.path.join(root, "scrfd_10g_bnkps.onnx")
        self.landmark_root = os.path.join(root, "2d106det.onnx")
        self.arcface_root = os.path.join(root, "glintr100.onnx")
        self.detection_model = SCRFD(self.detection_root, self.device)
        self.landmark_model = Landmark(self.landmark_root, self.device)
        self.arcface_model = ArcFace(self.arcface_root, self.device)
        
    def landmark_loss(self, id_landmark=None, gt_landmark=None, mask=None):
        # id_landmark: [B, F, 106, 2]
        # mask: [B, F]
        mask = mask.unsqueeze(-1).unsqueeze(-1) # [B, F] -> [B, F, 1, 1]
        error = torch.abs(id_landmark - gt_landmark) * mask
        valid_frame_count = mask.sum() + 1e-8  # 避免除零
        loss = error.sum() / valid_frame_count / id_landmark.shape[-2]
        return loss
    
    def embedding_loss(self, id_embedding=None, gt_embedding=None, mask=None):
        # edbedding: [B, F, C]
        # mask: [B, F]
        cos_sim = F.cosine_similarity(id_embedding, gt_embedding, dim=2) #[B, F, C]
        cos_loss = (1-cos_sim) * mask
        valid_frame_count = mask.sum() + 1e-8 # 避免除零
        loss = cos_loss.sum() / valid_frame_count
        return loss
    
    def pool_embedding_loss(self, id_embedding=None, gt_embedding=None, id_mask=None):
        # edbedding: [B, F, C]
        # mask: [B, F]
        id_emb_expanded = id_embedding.unsqueeze(2)
        gt_emb_expanded = gt_embedding.unsqueeze(1)
        gt_mask = torch.ones(gt_embedding.shape[0], gt_embedding.shape[1]).to(id_mask.device)
        gt_mask[:, 0] = 0
        is_all_zero = (gt_embedding == 0).all(dim=-1)
        gt_mask[is_all_zero] = 0

        cos_sim_all = F.cosine_similarity(id_emb_expanded, gt_emb_expanded, dim=3) 
        valid_mask = id_mask.unsqueeze(2) * gt_mask.unsqueeze(1)  # [B, F, F]

        gt_valid_count = gt_mask.sum(dim=1) + 1e-8
        weight_matrix = valid_mask / (gt_valid_count.unsqueeze(1).unsqueeze(2) + 1e-8)
        mean_similarities = (cos_sim_all * weight_matrix).sum(dim=2)
        # mean_similarities = cos_sim_all.mean(dim=2)
        cos_loss = mean_similarities * id_mask
        valid_frame_count = id_mask.sum() + 1e-8 # 避免除零
        loss = cos_loss.sum() / valid_frame_count
        return loss

if __name__ == '__main__':
    face_app = FaceAnalysis(root = "/data/models/antelopev2/",device='cuda')
    from decord import VideoReader
    import time
    import torch.nn.functional as F
    vr = VideoReader("./video1.mp4")
    print(len(vr))
    frames = [f.asnumpy() for f in vr]
    print(frames[0].shape, frames[0].dtype, frames[0].max(), frames[0].min())
    h, w = frames[0].shape[:2]
    id_landmark = []
    id_embedding = []
    id_mask = []
    index = 0
    index1 = 0
    all_start = time.time()
    for f in frames:
        f = torch.from_numpy(2*(f/255.)-1).permute(2,0,1).float().to('cuda') #(3, h, w)
        start = time.time()
        bboxes, kpss = face_app.detection_model.detect(f)
        end = time.time()
        index += end-start
        if bboxes.shape[0] > 0:
            indexed_bboxes = [(i, x) for i, x in enumerate(bboxes)]
            sorted_bboxes = sorted(indexed_bboxes, key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]))
            max_index, max_bbox = sorted_bboxes[-1]
            kps = kpss[max_index]
            start = time.time()
            face = Face(bbox=bboxes[max_index][0:4], kps=kps, det_score=bboxes[max_index][4])
            id_embedding.append(face_app.arcface_model.get(f, face))
            end = time.time()
            index1 += end-start
            id_mask.append(1)
        else:
            # id_landmark.append(torch.zeros(106, 2))
            id_embedding.append(torch.zeros(512))
            id_mask.append(0)
    all_end = time.time()
    print(all_end-all_start)
    print(index)
    id_embedding = torch.stack(id_embedding).unsqueeze(0)
    face_embeddings = torch.randn_like(id_embedding)
    id_mask = torch.tensor(id_mask).unsqueeze(0).to(id_embedding.device)
    face_score = face_app.pool_embedding_loss(id_embedding, face_embeddings, id_mask)
    
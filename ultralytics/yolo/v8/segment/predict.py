# Ultralytics YOLO 🚀, GPL-3.0 license

import torch
import pathlib
import vision6D as vis
import numpy as np
import cv2
import copy

from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.utils.plotting import colors, save_one_box, save_info, xywh2xyxy, pad_xyxy, clip_coords, scale_image
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

class PosePredictor(DetectionPredictor):
    
    def create_color_mask_gt(self, mask, image):
        mask = np.expand_dims(mask, axis=-1)
        self.app = vis.App(off_screen=True)
        self.app.set_transformation_matrix(self.gt_pose)
        self.app.load_meshes({'ossicles': self.ossicles_path})
        color_mask = self.app.plot()
        color_maskxseg_mask = color_mask * mask

        # make the color_mask channel order is the same as the image
        color_maskxseg_mask_bool = (0.2989 * color_maskxseg_mask[..., :1] + 0.5870*color_maskxseg_mask[..., 1:2] + 0.1140*color_maskxseg_mask[..., 2:]).astype("bool") 
        binary_maskxseg_mask = vis.utils.color2binary_mask(color_maskxseg_mask)
        assert (binary_maskxseg_mask == color_maskxseg_mask_bool).all(), "color_mask_binarized is not the same as the color_mask_bool"
        
        process_color_mask = np.where(binary_maskxseg_mask == 0, [1,1,1], color_maskxseg_mask[..., ::-1])
        process_image = np.where(binary_maskxseg_mask != 0, [1,1,1], image)
        render_image = process_color_mask * process_image
        render_image = render_image.astype('uint8')

        return color_mask, color_maskxseg_mask, binary_maskxseg_mask, render_image

    def crop_image(self, xyxy, im):
        xyxy = xyxy.int() if isinstance(xyxy, torch.Tensor) else xyxy.astype('int32')
        crop = im[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
        return crop

    """
    def create_2d_3d_pairs(self, color_mask, npts:int=-1, binary_mask=None):

        if binary_mask is None:
            binary_mask = vis.utils.color2binary_mask(color_mask)

        # To convert color_mask to bool type, we need to consider all three channels for color image, or conbine all channels to grey for color images!
        color_mask_bool = (0.2989 * color_mask[..., :1] + 0.5870*color_mask[..., 1:2] + 0.1140*color_mask[..., 2:]).astype("bool") 
        # # solution2
        # color_mask_bool = np.logical_or(color_mask.astype("bool")[..., :1], color_mask.astype("bool")[..., 1:2], color_mask.astype("bool")[..., 2:])
        # # solution3
        # color_mask_bool = color_mask.astype("bool")
        # color_mask_bool = (color_mask_bool[..., :1] + color_mask_bool[..., 1:2] + color_mask_bool[..., 2:]).astype("bool")
        assert (binary_mask == color_mask_bool).all(), "binary_mask is not the same as the color_mask_bool"

        # Randomly select points in the mask
        idx = np.where(binary_mask == 1)
        
        # swap the points for opencv, maybe because they handle RGB image differently (RGB -> BGR in opencv)
        # pts = np.array([(x,y) for x,y in zip(idx[1], idx[0])])
        x, y = idx[1], idx[0]
        pts = np.stack((x, y), axis=1)
        
        if npts == -1:
            rand_pts = pts
        else:
            rand_pts_idx = np.random.choice(pts.shape[0], npts)
            rand_pts = pts[rand_pts_idx,:]
            
        # # noise check
        # rand_pts = np.vstack((rand_pts, [0, 0]))
        
        # Obtain the 3D verticies (normaize rgb values)
        rgb = color_mask[rand_pts[:,1], rand_pts[:,0]]

        if np.max(rgb) > 1: rgb = rgb / 255

        ossicle_mesh = vis.utils.load_trimesh(self.ossicles_path)
        
        vertices = ossicle_mesh.vertices.T

        r = rgb[:, 0] * (np.max(vertices[0]) - np.min(vertices[0])) + np.min(vertices[0])
        g = rgb[:, 1] * (np.max(vertices[1]) - np.min(vertices[1])) + np.min(vertices[1])
        b = rgb[:, 2] * (np.max(vertices[2]) - np.min(vertices[2])) + np.min(vertices[2])
        vtx = np.stack([r, g, b], axis=1)
        
        return vtx, rand_pts
    """

    def create_2d_3d_pairs_xyxy_nocs(self, color_mask, binary_mask, xyxy):
        if isinstance(color_mask, np.ndarray):
            binary_mask = np.where(binary_mask != 0, 1, 0)
            binary_mask_bool = binary_mask.astype('bool')
            assert (binary_mask == binary_mask_bool).all(), "binary mask should be the same as binary mask bool"
            
            idx = np.where(binary_mask == 1)
            
            # swap the points for opencv, maybe because they handle RGB image differently (RGB -> BGR in opencv)
            x, y = idx[1], idx[0] # mask shape: (H, W, D)
            pts = np.stack((x, y), axis=1)

            xyxy = xyxy.int() if isinstance(xyxy, torch.Tensor) else xyxy.astype('int32')
            true_x, true_y = x + np.ones((x.shape)) * xyxy[0], y + np.ones((y.shape)) * xyxy[1]
            true_pts = np.stack((true_x.astype("int32"), true_y.astype("int32")), axis=1)
            
            # Obtain the 3D verticies (normalize rgb values)
            rgb = color_mask[pts[:,1], pts[:,0]] 
            
            if np.max(color_mask) > 1:
                rgb = rgb / 255

            ossicle_mesh = vis.utils.load_trimesh(self.ossicles_path)
            
            vertices = ossicle_mesh.vertices.T

            r = rgb[:, 0] * (np.max(vertices[0]) - np.min(vertices[0])) + np.min(vertices[0])
            g = rgb[:, 1] * (np.max(vertices[1]) - np.min(vertices[1])) + np.min(vertices[1])
            b = rgb[:, 2] * (np.max(vertices[2]) - np.min(vertices[2])) + np.min(vertices[2])
            vtx = np.stack([r, g, b], axis=1)

            # numpy implementation
            # vtx: [pts x c]
            # pts: [pts x c]

        # we take mask shape as [N, C, H, W] since the predicted masks' shape from torch is [N, C, H, W]
        elif isinstance(color_mask, torch.Tensor):
            idx = torch.where(binary_mask == 1)
            # swap the points for opencv, maybe because they handle RGB image differently (RGB -> BGR in opencv)
            # x, y = idx[2], idx[1] # mask shape: [N, H, W, C]
            x, y = idx[3], idx[2] # mask shape: [N, C, H, W]
            pts = torch.stack((x, y), axis=1)

            xyxy = xyxy.int() if isinstance(xyxy, torch.Tensor) else xyxy.astype('int32')
            true_x, true_y = x + (torch.ones((x.shape)) * xyxy[0]).cuda(), y + (torch.ones((y.shape)) * xyxy[1]).cuda()
            true_pts = torch.stack((true_x.int(), true_y.int()), axis=1)
                
            # Obtain the 3D verticies (normalize rgb values)
            # rgb = color_mask[:, pts[:,1], pts[:,0]] # mask shape: [N, H, W, C]
            rgb = color_mask[..., pts[:,1], pts[:,0]].permute(0,2,1) # mask shape: [N, C, H, W]
            
            if torch.max(color_mask) > 1:
                rgb = rgb / 255

            ossicle_mesh = vis.utils.load_trimesh(self.ossicles_path)
            
            vertices = torch.tensor(ossicle_mesh.vertices.T)

            r = rgb[..., 0] * (torch.max(vertices[0]) - torch.min(vertices[0])) + torch.min(vertices[0])
            g = rgb[..., 1] * (torch.max(vertices[1]) - torch.min(vertices[1])) + torch.min(vertices[1])
            b = rgb[..., 2] * (torch.max(vertices[2]) - torch.min(vertices[2])) + torch.min(vertices[2])
            vtx = torch.stack([r, g, b], dim=2).float()
            true_pts = true_pts.unsqueeze(0).float()

            # torch implementation
            # vtx: [b x pts x c]    
            # pts: [b x pts x c]

        return vtx, true_pts

    def create_2d_3d_pairs_xyxy_latlon(self, color_mask, binary_mask, xyxy):

        binary_mask = np.where(binary_mask != 0, 1, 0)
        binary_mask_bool = binary_mask.astype('bool')
        assert (binary_mask == binary_mask_bool).all(), "binary mask should be the same as binary mask bool"
        
        idx = np.where(binary_mask == 1)
        # swap the points for opencv, maybe because they handle RGB image differently (RGB -> BGR in opencv)
        idx = idx[:2][::-1]
        x, y = idx[0], idx[1]
        pts2d = np.stack((x, y), axis=1)

        # pad the pts2d to 1920x1080 from 640x640
        xyxy = xyxy.int() if isinstance(xyxy, torch.Tensor) else xyxy.astype('int32')
        true_x, true_y = x + np.ones((x.shape)) * xyxy[0], y + np.ones((y.shape)) * xyxy[1]
        pts2d_pad = np.stack((true_x.astype("int32"), true_y.astype("int32")), axis=1)

        pts3d = []
        
        # Obtain the rg color
        color = color_mask[pts2d[:,1], pts2d[:,0]][..., :2]
        if np.max(color) > 1: color = color / 255
        gx = color[:, 0]
        gy = color[:, 1]

        lat = np.array(self.app.latlon[..., 0])
        lon = np.array(self.app.latlon[..., 1])
        mesh = getattr(self.app, f'ossicles_mesh')
        lonf = lon[mesh.faces]
        msk = (np.sum(lonf>=0, axis=1)==3) & (np.sum(lat[mesh.faces]>=0, axis=1)==3)
        for i in range(len(pts2d)):
            pt = vis.utils.latLon2xyz(mesh, lat, lonf, msk, gx[i], gy[i])
            pts3d.append(pt)

        pts3d = np.array(pts3d).reshape((len(pts3d), 3))

        pts2d = pts2d_pad.astype('float32')
        pts3d = pts3d.astype('float32')

        return pts3d, pts2d

    def predict_pose(self, pts3d, pts2d):
        flag = True

        # test pnp algorithm
        camera_intrinsics = np.array([[5e+4, 0., 960.], [0., 5e+4, 540.], [0., 0., 1.]])
        camera_position = (0.0, 0.0, -500.0)
        predicted_pose = vis.utils.solve_epnp_cv2(pts2d, pts3d, camera_intrinsics, camera_position)

        try:
            assert np.isclose(predicted_pose, self.gt_pose, atol=10).all(), "predicted pose is not close to RT"
            self.predicted_pose = predicted_pose
        except AssertionError:
            flag = False

        return flag

class SegmentationPredictor(PosePredictor):

    def postprocess(self, preds, img, orig_imgs):
        # TODO: filter by classes
        p = ops.non_max_suppression(preds[0],
                                    self.args.conf,
                                    self.args.iou,
                                    agnostic=self.args.agnostic_nms,
                                    max_det=self.args.max_det,
                                    nc=len(self.model.names),
                                    classes=self.args.classes)
        results = []
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
        for i, pred in enumerate(p):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            path, _, _, _, _ = self.batch
            img_path = path[i] if isinstance(path, list) else path
            if not len(pred):  # save empty boxes
                results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6]))
                continue
            if self.args.retina_masks:
                if not isinstance(orig_imgs, torch.Tensor):
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                if not isinstance(orig_imgs, torch.Tensor):
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(
                Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))
        return results

    def write_results(self, idx, results, batch):
        p, im, im0 = batch
        log_string = ''

        self.previous_center = self.center if self.args.save_poses else []

        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        imc = copy.deepcopy(im0)
        if self.source_type.webcam or self.source_type.from_img:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        result = results[idx]
        if len(result) == 0:
            return f'{log_string}(no detections), '
        det, mask = result.boxes, result.masks  # getting tensors TODO: mask mask,box inherit for tensor

        # detect more than one result per frame
        if len(det) >= 2:
            det = det[0]
        if len(mask) >= 2:
            mask.masks = mask.masks[0][None]
            # mask.segments = [mask.segments[0]] # error: cannot set attributes
    
        # make a square box label
        det.xywh[0][2] = 640.0
        det.xywh[0][3] = 640.0
        xyxy = xywh2xyxy(det.xywh.view(1, 4)).int()
        # clip boxes, just incase the box is moving out the image
        clip_coords(xyxy, imc.shape)
        xyxy = pad_xyxy(xyxy, (640, 640))
        det.xyxy[0][0], det.xyxy[0][1], det.xyxy[0][2], det.xyxy[0][3] = xyxy[0][0], xyxy[0][1], xyxy[0][2], xyxy[0][3]

        # Print results
        for c in det.cls.unique():
            n = (det.cls == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        # Mask plotting
        if self.args.save or self.args.show:
            im_gpu = torch.as_tensor(im0, dtype=torch.float16, device=mask.masks.device).permute(
                2, 0, 1).flip(0).contiguous() / 255 if self.args.retina_masks else im[idx]
            self.annotator.masks(masks=mask.masks, colors=[colors(x, True) for x in det.cls], im_gpu=im_gpu)

        # Write results
        for j, d in enumerate(reversed(det)):
            assert j == 0 # only one detection and segmentation per frame
            c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
            if self.args.save_txt:  # Write to file
                if len(mask.masks) != 0:
                    seg_points = mask.xyn[len(det) - j - 1].copy().reshape(-1) # save the segmentation points
                    original_masks = mask.masks.clone().permute(1,2,0).detach().cpu().numpy()
                    original_masks = scale_image(original_masks.shape, original_masks, (1080, 1920, 1))
                    (h, w, _) = np.where(original_masks == 1)
                    seg_mask = np.zeros((1080, 1920))
                    seg_mask[h, w] = 1
                    seg_mask = seg_mask.astype('bool')

                    if not self.args.save_poses:
                        bbox_xyxy = det.xyxy.clone().cpu().numpy().reshape(-1)
                        bbox_xywh = det.xywh.clone().cpu().numpy().reshape(-1)

                        name = pathlib.Path(self.txt_path + ".json").stem
                        self.info_container[name] = {}
                        self.info_container[name]['seg'] = seg_points.tolist()
                        self.info_container[name]['bbox_xyxy'] = bbox_xyxy.tolist()
                        self.info_container[name]['bbox_xywh'] = bbox_xywh.tolist()
                else:
                    seg_mask = None
       
            if self.args.save or self.args.show:  # Add bbox to image
                name = ('' if id is None else f'id:{id} ') + self.model.names[c]
                label = None if self.args.hide_labels else (name if self.args.hide_conf else f'{name} {conf:.2f}')
                if self.args.boxes:
                    self.annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))

            if self.args.save_crop:
                save_one_box(d.xyxy, imc, file=self.save_dir / 'crops' / self.model.names[c] / f'{self.data_path.stem}.jpg')

            if self.args.save_poses:
                # save data to the corresponding container if the segmentation mask exists
                if seg_mask is not None:
                    # use connected component to filter out the noise
                    res = cv2.connectedComponents(seg_mask.astype('uint8'))
                    if res[0] > 2: seg_mask = (res[1] == 1)

                    self.center = (int(det.xywh[0, 0]), int(det.xywh[0, 1]))

                    if len(self.previous_center) == 0: self.previous_center = self.center
                    
                    tx_gt_pose = self.center[0] - self.previous_center[0]
                    ty_gt_pose = self.center[1] - self.previous_center[1]

                    # get the new gt_pose
                    self.gt_pose[0, 3] = self.gt_pose[0, 3] + 0.01*tx_gt_pose
                    self.gt_pose[1, 3] = self.gt_pose[1, 3] + 0.01*ty_gt_pose

                    color_mask, color_maskxseg_mask, binary_maskxseg_mask, self.render_image = self.create_color_mask_gt(seg_mask, imc)
                    
                    flag = False

                    # Use the mask x segmented mask
                    if np.sum(color_maskxseg_mask) != 0:
                        color_mask_crop = self.crop_image(det.xyxy.view(-1), color_maskxseg_mask)
                        binary_mask_crop = self.crop_image(det.xyxy.view(-1), binary_maskxseg_mask)
                        render_image_crop = self.crop_image(det.xyxy.view(-1), self.render_image)

                        if self.args.rescale_ratio != 1:
                            # downscale the cropped color mask, color_mask_crop shape is (640, 640, 3)
                            color_mask_crop_downscale = cv2.resize(color_mask_crop, (int(color_mask_crop.shape[0] * self.args.rescale_ratio), int(color_mask_crop.shape[1] * self.args.rescale_ratio)), interpolation = cv2.INTER_AREA) 
                            binary_mask_crop_downscale = cv2.resize(binary_mask_crop, (int(color_mask_crop.shape[0] * self.args.rescale_ratio), int(color_mask_crop.shape[1] * self.args.rescale_ratio)), interpolation = cv2.INTER_AREA)
                            # upscale the cropped color mask to the original size
                            color_mask_crop = cv2.resize(color_mask_crop_downscale, (640, 640), interpolation = cv2.INTER_AREA)
                            binary_mask_crop = cv2.resize(binary_mask_crop_downscale, (640, 640), interpolation = cv2.INTER_AREA)

                        if self.app.nocs_color:   
                            # test the cropped and segmented mask
                            p3d, p2d = self.create_2d_3d_pairs_xyxy_nocs(color_mask_crop, binary_mask_crop, det.xyxy.view(-1).cpu().numpy())
                        else:
                            p3d, p2d = self.create_2d_3d_pairs_xyxy_latlon(color_mask_crop, binary_mask_crop, det.xyxy.view(-1).cpu().numpy())

                        flag = self.predict_pose(pts3d=p3d, pts2d=p2d)
                        
                    if flag:
                        name = save_info(image=imc[..., ::-1], image_file = self.save_dir / 'images' / self.model.model.names[c] / f'{self.data_path.stem}.png')
                        color_mask_name = save_info(image=color_mask, image_file = self.save_dir / 'color_masks' / self.model.model.names[c] / f'{self.data_path.stem}.png')
                        seg_mask_name = save_info(image=seg_mask, image_file = self.save_dir / 'seg_masks' / self.model.model.names[c] / f'{self.data_path.stem}.png')    
                        color_seg_mask_name = save_info(image=color_maskxseg_mask, image_file = self.save_dir / 'color_seg_masks' / self.model.model.names[c] / f'{self.data_path.stem}.png')
                        render_image_name = save_info(image=render_image_crop[..., ::-1], image_file = self.save_dir / 'render_images_crop' / self.model.model.names[c] / f'{self.data_path.stem}.png')    
                        assert name == color_mask_name == seg_mask_name == color_seg_mask_name == render_image_name, "imc name is not the same as the color_mask / render_image_crop name!"
                        
                        self.info_container[name] = {}
                        self.info_container[name]['xyxy'] = det.xyxy.view(-1).tolist()
                        self.info_container[name]['xywh'] = det.xywh.view(-1).tolist()
                        self.info_container[name]['seg'] = seg_points.tolist()
                        self.info_container[name]['gt_pose'] = self.gt_pose.tolist()

                    else:
                        self.render_image = None

                else:
                    self.render_image = None

        return log_string

def predict(cfg=DEFAULT_CFG, use_python=False):
    model = cfg.model or 'yolov8n-seg.pt'
    source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
        else 'https://ultralytics.com/images/bus.jpg'

    args = dict(model=model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = SegmentationPredictor(overrides=args)
        predictor.predict_cli()

if __name__ == '__main__':
    predict()

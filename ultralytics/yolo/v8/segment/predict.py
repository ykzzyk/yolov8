# Ultralytics YOLO 🚀, GPL-3.0 license

import torch
import pathlib

from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.utils.plotting import colors, save_one_box, xywh2xyxy, pad_xyxy, clip_coords
from ultralytics.yolo.v8.detect.predict import DetectionPredictor


class SegmentationPredictor(DetectionPredictor):

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
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        imc = im0.copy() if self.args.save_crop else im0
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
            c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
            if self.args.save_txt:  # Write to file
                seg = mask.xyn[len(det) - j - 1].copy().reshape(-1)  # reversed mask.xyn, (n,2) to (n*2)
                bbox_xyxy = det.xyxy.clone().cpu().numpy().reshape(-1)
                bbox_xywh = det.xywh.clone().cpu().numpy().reshape(-1)

                name = pathlib.Path(self.txt_path + ".json").stem
                self.info_container[name] = {}
                self.info_container[name]['seg'] = seg.tolist()
                self.info_container[name]['bbox_xyxy'] = bbox_xyxy.tolist()
                self.info_container[name]['bbox_xywh'] = bbox_xywh.tolist()
       
            if self.args.save or self.args.show:  # Add bbox to image
                name = ('' if id is None else f'id:{id} ') + self.model.names[c]
                label = None if self.args.hide_labels else (name if self.args.hide_conf else f'{name} {conf:.2f}')
                if self.args.boxes:
                    self.annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))
            if self.args.save_crop:
                save_one_box(d.xyxy, imc, file=self.save_dir / 'crops' / self.model.names[c] / f'{self.data_path.stem}.jpg')

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

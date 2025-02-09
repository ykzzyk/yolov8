# Ultralytics YOLO 🚀, GPL-3.0 license
"""
Run prediction on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ yolo mode=predict model=yolov8n.pt source=0                               # webcam
                                                img.jpg                         # image
                                                vid.mp4                         # video
                                                screen                          # screenshot
                                                path/                           # directory
                                                list.txt                        # list of images
                                                list.streams                    # list of streams
                                                'path/*.jpg'                    # glob
                                                'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ yolo mode=predict model=yolov8n.pt                 # PyTorch
                              yolov8n.torchscript        # TorchScript
                              yolov8n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                              yolov8n_openvino_model     # OpenVINO
                              yolov8n.engine             # TensorRT
                              yolov8n.mlmodel            # CoreML (macOS-only)
                              yolov8n_saved_model        # TensorFlow SavedModel
                              yolov8n.pb                 # TensorFlow GraphDef
                              yolov8n.tflite             # TensorFlow Lite
                              yolov8n_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov8n_paddle_model       # PaddlePaddle
"""
import platform
from collections import defaultdict
from pathlib import Path

import cv2
import copy
import numpy as np
import json
import pathlib
import vision6D as vis
import os

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data import load_inference_source
from ultralytics.yolo.data.augment import classify_transforms
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from ultralytics.yolo.utils.checks import check_imgsz, check_imshow
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.torch_utils import select_device, smart_inference_mode

STREAM_WARNING = """
    WARNING ⚠️ stream/video/webcam/dir predict source will accumulate results in RAM unless `stream=True` is passed,
    causing potential out-of-memory errors for large sources or long-running streams/videos.

    Usage:
        results = model(source=..., stream=True)  # generator of Results objects
        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            masks = r.masks  # Masks object for segment masks outputs
            probs = r.probs  # Class probabilities for classification outputs
"""

class CustomPredictor:
    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        """
        Initializes the CustomPredictor class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """

        self.args = get_cfg(cfg, overrides)

    def load_info(self, pathname):

        id = pathname.split('.')[1]
        side = self.side
        config = vis.config

        ossicles_path = getattr(config, f"OSSICLES_MESH_PATH_{id}_{side}")
        facial_nerve_path = getattr(config, f"FACIAL_NERVE_MESH_PATH_{id}_{side}")
        chorda_path = getattr(config, f"CHORDA_MESH_PATH_{id}_{side}")
        scala_tympani_path = getattr(config, f"SCALA_TYMPANI_MESH_PATH_{id}_{side}")

        with open(str(self.save_dir / "meshes" / "mesh_path.txt"), 'w') as f:
            f.write(str(ossicles_path) + '\n')
            f.write(str(facial_nerve_path) + '\n')
            f.write(str(chorda_path) + '\n')
            f.write(str(scala_tympani_path) + '\n')

        gt_pose_path = getattr(config, f"gt_pose_{id}_{side}")

        return ossicles_path, facial_nerve_path, chorda_path, scala_tympani_path, gt_pose_path

    def project_pose_cpu(self, image_source, predicted_pose, ossicles_path, facial_nerve_path, chorda_path, scala_tympani_path):
        app = vis.App(off_screen=True, nocs_color=False)
        app.set_transformation_matrix(predicted_pose)
        app.load_image(image_source)
        app.load_meshes({'ossicles': ossicles_path, 'facial_nerve': facial_nerve_path, 'chorda': chorda_path, 'scala_tympani': scala_tympani_path})
        app.set_reference("ossicles")
        # set the opacity attributes since the default is not opaque
        app.set_image_opacity(0.99)
        app.set_mesh_opacity(0.8)
        render_image = app.plot()
        return render_image

class BasePredictor(CustomPredictor):
    """
    BasePredictor

    A base class for creating predictors.

    Attributes:
        args (SimpleNamespace): Configuration for the predictor.
        save_dir (Path): Directory to save results.
        done_setup (bool): Whether the predictor has finished setup.
        model (nn.Module): Model used for prediction.
        data (dict): Data configuration.
        device (torch.device): Device used for prediction.
        dataset (Dataset): Dataset used for prediction.
        vid_path (str): Path to video file.
        vid_writer (cv2.VideoWriter): Video writer for saving video output.
        annotator (Annotator): Annotator used for prediction.
        data_path (str): Path to data.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None):

        super().__init__()
        
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = self.args.name or f'{self.args.mode}'
        self.save_dir = increment_path(Path(project) / name, exist_ok=self.args.exist_ok)
        os.makedirs(self.save_dir / "videos", exist_ok=True)
        os.makedirs(self.save_dir / "meshes", exist_ok=True)
        if self.args.conf is None:
            self.args.conf = 0.5  # default conf=0.5
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # Usable if setup is done
        self.model = None
        self.side = self.args.side # set the side of the ossicles
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.annotator = None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.callbacks = defaultdict(list, callbacks.default_callbacks)  # add callbacks
        callbacks.add_integration_callbacks(self)

    def preprocess(self, img):
        pass

    def get_annotator(self, img):
        raise NotImplementedError('get_annotator function needs to be implemented')

    def write_results(self, results, batch, print_string):
        raise NotImplementedError('print_results function needs to be implemented')

    def postprocess(self, preds, img, orig_img):
        return preds

    def __call__(self, source=None, model=None, stream=False):
        self.stream = stream
        if stream:
            return self.stream_inference(source, model)
        else:
            return list(self.stream_inference(source, model))  # merge list of Result into one

    def predict_cli(self, source=None, model=None):
        # Method used for CLI prediction. It uses always generator as outputs as not required by CLI mode
        gen = self.stream_inference(source, model)
        for _ in gen:  # running CLI inference without accumulating any outputs (do not modify)
            pass

    def setup_source(self, source):
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        if self.args.task == 'classify':
            transforms = getattr(self.model.model, 'transforms', classify_transforms(self.imgsz[0]))
        else:  # predict, segment
            transforms = None
        self.dataset = load_inference_source(source=source,
                                             transforms=transforms,
                                             imgsz=self.imgsz,
                                             vid_stride=self.args.vid_stride,
                                             stride=self.model.stride,
                                             auto=self.model.pt)
        self.source_type = self.dataset.source_type
        if not getattr(self, 'stream', True) and (self.dataset.mode == 'stream' or  # streams
                                                  len(self.dataset) > 1000 or  # images
                                                  any(getattr(self.dataset, 'video_flag', [False]))):  # videos
            LOGGER.warning(STREAM_WARNING)
        
        self.vid_path, self.vid_writer = [None] * self.dataset.bs, [None] * self.dataset.bs
        self.projection_vid_path, self.projection_vid_writer = [None] * self.dataset.bs, [None] * self.dataset.bs
        self.render_image_vid_path, self.render_image_vid_writer = [None] * self.dataset.bs, [None] * self.dataset.bs

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None):
        if self.args.verbose:
            LOGGER.info('')

        # setup model
        if not self.model:
            self.setup_model(model)
        # setup source every time predict is called
        self.setup_source(source if source is not None else self.args.source)

        # warmup model
        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
            self.done_warmup = True

        self.seen, self.windows, self.dt, self.batch = 0, [], (ops.Profile(), ops.Profile(), ops.Profile()), None
        self.run_callbacks('on_predict_start')

        # path = pathlib.Path(self.dataset.files[0])
        # ossicles_path, facial_nerve_path, chorda_path, scala_tympani_path, gt_pose = self.load_info(path.stem)

        if self.args.save_poses:
            path = pathlib.Path(self.dataset.files[0])

            # load the info needed for projection
            ossicles_path, facial_nerve_path, chorda_path, scala_tympani_path, gt_pose = self.load_info(path.stem)
            self.ossicles_path = ossicles_path
            self.gt_pose = gt_pose

            projection_save_path = str(self.save_dir / "videos" / path.stem) + '_projection.mp4'
            render_image_save_path = str(self.save_dir / "videos" / path.stem) + '_render_image.mp4'
            self.center = []
            self.render_image = None

        for batch in self.dataset:
            self.run_callbacks('on_predict_batch_start')
            self.batch = batch
            path, im, im0s, vid_cap, s = batch
            visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.args.visualize else False

            # preprocess
            with self.dt[0]:
                im = self.preprocess(im)
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # inference
            with self.dt[1]:
                preds = self.model(im, augment=self.args.augment, visualize=visualize)

            # postprocess
            with self.dt[2]:
                self.results = self.postprocess(preds, im, im0s)
            self.run_callbacks('on_predict_postprocess_end')

            # visualize, save, write results
            n = len(im)
            for i in range(n):
                self.results[i].speed = {
                    'preprocess': self.dt[0].dt * 1E3 / n,
                    'inference': self.dt[1].dt * 1E3 / n,
                    'postprocess': self.dt[2].dt * 1E3 / n}
                if self.source_type.tensor:  # skip write, show and plot operations if input is raw tensor
                    continue
                p, im0 = (path[i], im0s[i].copy()) if self.source_type.webcam or self.source_type.from_img \
                    else (path, im0s.copy())
                p = Path(p)
                imc = copy.deepcopy(im0)

                if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                    s += self.write_results(i, self.results, (p, im, im0))

                if self.args.show:
                    self.show(p)

                if self.args.save:
                    res = self.annotator.result()
                    self.save_preds(res, vid_cap, i, str(self.save_dir / "videos" / p.name), self.vid_path, self.vid_writer)

                if self.args.save_poses:
                    if self.render_image is not None:
                        project_image = self.project_pose_cpu(imc[...,::-1], self.predicted_pose, ossicles_path, facial_nerve_path, chorda_path, scala_tympani_path)
                        # convert project_image RGB to BGR
                        self.save_preds(project_image[..., ::-1], vid_cap, i, projection_save_path, self.projection_vid_path, self.projection_vid_writer)
                        # self.render_image channel order is BGR, no need to convert
                        self.save_preds(self.render_image, vid_cap, i, render_image_save_path, self.render_image_vid_path, self.render_image_vid_writer)

            self.run_callbacks('on_predict_batch_end')
            yield from self.results

            # Print time (inference-only)
            if self.args.verbose:
                LOGGER.info(f'{s}{self.dt[1].dt * 1E3:.1f}ms')

        # Release assets
        if isinstance(self.vid_writer[-1], cv2.VideoWriter):
            self.vid_writer[-1].release()  # release final video writer

        # Print results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1E3 for x in self.dt)  # speeds per image
            LOGGER.info(f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape '
                        f'{(1, 3, *self.imgsz)}' % t)
        
        self.run_callbacks('on_predict_end')

    def setup_model(self, model, verbose=True):
        device = select_device(self.args.device, verbose=verbose)
        model = model or self.args.model
        self.args.half &= device.type != 'cpu'  # half precision only supported on CUDA
        self.model = AutoBackend(model,
                                 device=device,
                                 dnn=self.args.dnn,
                                 data=self.args.data,
                                 fp16=self.args.half,
                                 verbose=verbose)
        self.device = device
        self.model.eval()

    def show(self, p):
        im0 = self.annotator.result()
        if platform.system() == 'Linux' and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
        cv2.imshow(str(p), im0)
        cv2.waitKey(500 if self.batch[4].startswith('image') else 1)  # 1 millisecond
    
    def save_preds(self, image, vid_cap, idx, save_path, vid_path, vid_writer):
        # save imgs
        if self.dataset.mode == 'image':
            cv2.imwrite(save_path, image)
        else:  # 'video' or 'stream'
            if vid_path[idx] != save_path:  # new video
                vid_path[idx] = save_path
                if isinstance(vid_writer[idx], cv2.VideoWriter):
                    vid_writer[idx].release()  # release previous video writer
                if vid_cap:  # video
                    fps = int(vid_cap.get(cv2.CAP_PROP_FPS))  # integer required, floats produce error in MP4 codec
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, image.shape[1], image.shape[0]
                save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                vid_writer[idx] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer[idx].write(image)

    def run_callbacks(self, event: str):
        for callback in self.callbacks.get(event, []):
            callback(self)

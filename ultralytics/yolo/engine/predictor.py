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

GIT_ROOT = pathlib.Path(os.path.abspath(__file__)).parent.parent.parent.parent.parent

class BasePredictor:
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
        """
        Initializes the BasePredictor class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = self.args.name or f'{self.args.mode}'
        self.save_dir = increment_path(Path(project) / name, exist_ok=self.args.exist_ok)
        if self.args.conf is None:
            self.args.conf = 0.5  # default conf=0.5
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # store the information for segmentation mask and two representation of BBox
        self.info_container = defaultdict(dict)
        
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

    
    def project_pose_cpu(self, image_source, predicted_pose, ossicles_path, facial_nerve_path, chorda_path):
        app = vis.App(off_screen=True)
        app.load_image(image_source)
        app.set_transformation_matrix(predicted_pose)
        app.set_image_opacity(0.8)
        app.set_mesh_opacity(0.999)
        app.load_meshes({'ossicles': ossicles_path, 'facial_nerve': facial_nerve_path, 'chorda': chorda_path})
        app.bind_meshes("ossicles", "g")
        app.bind_meshes("chorda", "h")
        app.bind_meshes("facial_nerve", "j")
        app.set_reference("ossicles")
        render_image = app.plot()

        return render_image
    
    def transform_vertices(self, vertices, r, t):
        transformed_vertices = r.reshape(1, 3, 3) @ vertices.reshape(vertices.shape[0], 3, 1) + t.reshape(1, 3, 1)
        return transformed_vertices.squeeze(-1) # make sure output shape is (B, P, 3)
    
    """
    def project_pose_gpu(self, frame, predicted_pose, ossicles_path, facial_nerve_path, chorda_path):

        # Setup
        if torch.cuda.is_available(): 
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else: device = torch.device("cpu")
        
        meshobjs = []
        for meshpath in [ossicles_path, facial_nerve_path, chorda_path]:
            with open(meshpath, "rb") as fid: mesh = vis.utils.meshread(fid)
            orient = mesh.orient / np.array([1,2,3])
            mesh.vertices = mesh.vertices * np.expand_dims(mesh.sz, axis=1) * np.expand_dims(orient, axis=1)
            verts_rgb_colors = torch.tensor(vis.utils.color_mesh(mesh.vertices.T)).float().unsqueeze(0).cuda()

            # de-couple the pose, and use rot and t seperately! 
            flip_xy_rot = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
            rot = predicted_pose[:3, :3]
            t = predicted_pose[:3, -1].reshape((-1, 1))
            mesh.vertices = self.transform_vertices(mesh.vertices.T, flip_xy_rot@rot, flip_xy_rot@t)
            
            verts = torch.tensor(mesh.vertices).float().unsqueeze(0).cuda()
            faces = torch.tensor(mesh.triangles.T).float().unsqueeze(0).cuda()
            textures = pytorch3d.renderer.TexturesVertex(verts_features=verts_rgb_colors)
            mesh = pytorch3d.structures.Meshes(verts=verts, faces=faces, textures=textures)
            meshobjs.append(mesh)
            
        # Join all the meshes as a scene
        meshes = pytorch3d.structures.join_meshes_as_scene(meshobjs, include_textures=True)

        # Initialize a camera.
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(-500, 0, 0, device=device)
        cameras = pytorch3d.renderer.cameras.FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.01, zfar=1000.01, fov=1.2375407233957483, degrees=True) # values from pyvista
        
        # Define the settings for rasterization and shading. 
        raster_settings = pytorch3d.renderer.RasterizationSettings(image_size=(1080, 1920))

        # Create a Phong renderer by composing a rasterizer and a shader.
        # pytorch3d.renderer.blending.BlendParams.background_color = (1, 1, 1) # default background color white
        pytorch3d.renderer.blending.BlendParams.sigma = 0.8
        pytorch3d.renderer.blending.BlendParams.gamma = 0.5
        
        # initial the lights to be None by setting it to 1 and 0
        lights = pytorch3d.renderer.lighting.DirectionalLights(ambient_color=((1, 1, 1), ), diffuse_color=((0, 0, 0), ), specular_color=((0, 0, 0), ), direction=((0, 0, 0), ), device=device)
        
        # materials = pytorch3d.renderer.Materials(device=device, ambient_color=((1, 1, 1), ), diffuse_color=((1, 1, 1), ), specular_color=((1, 1, 1), ), shininess=0)

        # create the renderer
        renderer = pytorch3d.renderer.MeshRenderer(rasterizer=pytorch3d.renderer.MeshRasterizer(raster_settings=raster_settings),
                                                shader=pytorch3d.renderer.SoftPhongShader(device=device, lights=lights)) #, materials=materials))

        # render images given the camera parameters
        res = renderer(meshes, cameras=cameras)
        
        image = res[0, ..., :3].cpu().numpy()
        
        render_image = (frame * image).astype('uint8')
        
        # # the rendered mesh is bright, but we cannot see through the rendered objects
        # frame = np.where(scene != [1, 1, 1], [1, 1, 1], frame/255) # make the frame ranges from [0, 1]
        # rendered_scene = ((frame * scene) * 255).astype('uint8')

        return render_image
    """

    def load_info(self, pathname):

        identifier = pathname.split('.')[1]

        ossicles_path = GIT_ROOT / "ossicles_6D_pose_estimation" / "data" / "surgical_planning" / pathname / "mesh" / f"{identifier}_{self.side}_ossicles_processed.mesh"
        facial_nerve_path = GIT_ROOT / "ossicles_6D_pose_estimation" / "data" / "surgical_planning" / pathname / "mesh" / f"{identifier}_{self.side}_facial_nerve_processed.mesh"
        chorda_path = GIT_ROOT / "ossicles_6D_pose_estimation" / "data" / "surgical_planning" / pathname / "mesh" / f"{identifier}_{self.side}_chorda_processed.mesh"
        gt_pose_path = GIT_ROOT / "ossicles_6D_pose_estimation" / "data" / "gt_poses" / f"{identifier}_{self.side}_gt_pose.npy"
        
        return ossicles_path, facial_nerve_path, chorda_path, gt_pose_path

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

        # check if save_dir/ label file exists
        if self.args.save or self.args.save_txt:
            (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        # warmup model
        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
            self.done_warmup = True

        self.seen, self.windows, self.dt, self.batch = 0, [], (ops.Profile(), ops.Profile(), ops.Profile()), None
        self.run_callbacks('on_predict_start')

        if self.args.save_poses:
            path = pathlib.Path(self.dataset.files[0])

            # load the info needed for projection
            ossicles_path, facial_nerve_path, chorda_path, gt_pose_path = self.load_info(path.stem)
            self.ossicles_path = ossicles_path

            # convert the gt pose path to numpy array
            self.gt_pose = np.load(gt_pose_path)

            projection_save_path = str(self.save_dir / path.stem) + '_projection.mp4'
            render_image_save_path = str(self.save_dir / path.stem) + '_render_image.mp4'
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
                    self.save_preds(res, vid_cap, i, str(self.save_dir / p.name), self.vid_path, self.vid_writer)

                if self.args.save_poses:
                    if self.render_image is not None:
                        project_image = self.project_pose_cpu(imc[...,::-1], self.gt_pose, ossicles_path, facial_nerve_path, chorda_path)
                        # convert project_image RGB to BGR
                        self.save_preds(project_image[..., ::-1], vid_cap, i, projection_save_path, self.projection_vid_path, self.projection_vid_writer)
                        # self.render_image channel order is BGR, no need to convert
                        self.save_preds(self.render_image, vid_cap, i, render_image_save_path, self.render_image_vid_path, self.render_image_vid_writer)

            self.run_callbacks('on_predict_batch_end')
            yield from self.results

            # Print time (inference-only)
            if self.args.verbose:
                LOGGER.info(f'{s}{self.dt[1].dt * 1E3:.1f}ms')

        if self.args.save:
            with open(pathlib.Path(self.txt_path).parent / "info.json", "w") as f:
                json.dump(self.info_container, f, indent=4)

        # Release assets
        if isinstance(self.vid_writer[-1], cv2.VideoWriter):
            self.vid_writer[-1].release()  # release final video writer

        # Print results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1E3 for x in self.dt)  # speeds per image
            LOGGER.info(f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape '
                        f'{(1, 3, *self.imgsz)}' % t)
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob('labels/*.txt')))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")

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
    """
    def save_preds(self, vid_cap, idx, save_path):
        im0 = self.annotator.result()
        # save imgs
        if self.dataset.mode == 'image':
            cv2.imwrite(save_path, im0)
        else:  # 'video' or 'stream'
            if self.vid_path[idx] != save_path:  # new video
                self.vid_path[idx] = save_path
                if isinstance(self.vid_writer[idx], cv2.VideoWriter):
                    self.vid_writer[idx].release()  # release previous video writer
                if vid_cap:  # video
                    fps = int(vid_cap.get(cv2.CAP_PROP_FPS))  # integer required, floats produce error in MP4 codec
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                self.vid_writer[idx] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            self.vid_writer[idx].write(im0)
    """
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

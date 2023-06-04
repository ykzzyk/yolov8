# line_seg = seg + (conf, ) * self.args.save_conf + (() if id is None else (id, ))
# with open(f'{self.txt_path + "_seg"}.txt', 'a') as f:
#     f.write(('%g ' * len(seg)).rstrip() % (seg) + '\n')
with open(f'{self.txt_path + "_seg"}.json', 'a') as f:
    json.dumps(seg.tolist())

bbox_xyxy = det.xyxy.clone().cpu().numpy().reshape(-1)
with open(f'{self.txt_path + "_bbox_xyxy"}.txt', 'a') as f:
    f.write(('%g ' * len(bbox_xyxy)).rstrip() % (bbox_xyxy) + '\n')

bbox_xywh = det.xywh.clone().cpu().numpy().reshape(-1)
with open(f'{self.txt_path + "_bbox_xywh"}.txt', 'a') as f:
    f.write(('%g ' * len(bbox_xywh)).rstrip() % (bbox_xywh) + '\n')

#^ predictor file
"""
    def transform_vertices(self, vertices, r, t):
        transformed_vertices = r.reshape(1, 3, 3) @ vertices.reshape(vertices.shape[0], 3, 1) + t.reshape(1, 3, 1)
        return transformed_vertices.squeeze(-1) # make sure output shape is (B, P, 3)
    
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

# task = Task.init(project_name='YOLOv8', task_name='ossicles_bbox_segmentation')
# pt = torch.load(root / "last.pt")
# pt['train_args']['epochs'] = 2000
# torch.save(pt, root / "last.pt")


"""
class Pose6D():
    def __init__(self, meshpaths, gt_pose_path, window_size=(1920, 1080), scale_factor=[0.01,0.01,1], image_opacity=0.999, surface_opacity=0.999, cam_focal_length=50000, cam_viewup=(0,-1,0)):
        self.meshpaths = meshpaths
        self.gt_pose = np.load(gt_pose_path)
        self.window_size = window_size
        self.scale_factor = scale_factor
        self.image_opacity = image_opacity
        self.surface_opacity = surface_opacity
        
        self.data = {}
        
        self.camera = pv.Camera()
        self.cam_focal_length = cam_focal_length
        self.cam_viewup = cam_viewup
        self.cam_position = -(self.cam_focal_length/100) # -500mm
        
        # set camera intrinsics and extrinsics
        self.set_camera_intrinsics()
        self.set_camera_extrinsics()
    
    def set_camera_extrinsics(self):
        self.camera.SetPosition((0,0,self.cam_position))
        self.camera.SetFocalPoint((0,0,0))
        self.camera.SetViewUp(self.cam_viewup)
    
    def set_camera_intrinsics(self): 
        # Set camera intrinsic attribute
        self.camera.SetWindowCenter(0, 0) # set window center
        view_angle = (180 / math.pi) * (2.0 * math.atan2(self.window_size[1]/2.0, self.cam_focal_length)) # or view_angle = np.degrees(2.0 * math.atan2(height/2.0, f))
        self.camera.SetViewAngle(view_angle) # set view angle should be in degrees
        
    def project_pose_cpu(self, frame):
        
        # Set the camera initial parameters
        self.pv_plotter = pv.Plotter(window_size=self.window_size, off_screen=True)
        self.pv_plotter.camera = self.camera.copy()
        self.pv_plotter.enable_joystick_actor_style()
        
        self.data['image'] = pv.UniformGrid(dimensions=(1920, 1080, 1), spacing=self.scale_factor, origin=(0.0, 0.0, 0.0))
        self.data['image'].point_data["values"] = frame.reshape((1920*1080, 3)) # order = 'C
        self.data['image'] = self.data['image'].translate(-1 * np.array(self.data['image'].center), inplace=False)
            
        # Then add it to the plotter
        image = self.pv_plotter.add_mesh(self.data['image'], rgb=True, opacity=self.image_opacity, name='image')
        self.pv_plotter.add_actor(image, pickable=False, name="image")

        for mesh_name, mesh_source in self.meshpaths.items():
        
            trimesh_data = vis.utils.load_trimesh(mesh_source)
            mesh_data = pv.wrap(trimesh_data)

            # Save the mesh data to dictionary
            self.data[mesh_name] = mesh_data

            # Color the vertex: set the color to be the meshes' initial location, and never change the color
            colors = vis.utils.color_mesh(mesh_data.points)
            mesh_data.point_data.set_scalars(colors)
            mesh = self.pv_plotter.add_mesh(mesh_data, rgb=True, opacity=self.surface_opacity, name=mesh_name) #, show_edges=True)
            # Set the transformation matrix to be the mesh's user_matrix
            mesh.user_matrix = self.gt_pose
            # Add and save the actor
            self.pv_plotter.add_actor(mesh, pickable=True, name=mesh_name)
        
        self.pv_plotter.disable()
        self.pv_plotter.show()
        frame = self.pv_plotter.last_image
        return frame
     
    # def project_pose_gpu(self, frame):
    #     frame = vis.utils.pytorch3d_render(self.meshpaths, self.gt_pose, frame)
    #     return frame 

class VideoProcessor():
    def __init__(self, pose6d):
        self.pose6d = pose6d
        self.previous_center = []
        model_path = root.parent / "yolov8" / "best.pt"
        self.model = YOLO(model_path)
        
    def yolo_predict(self, img, conf=0.8, box_size=(640, 640)):
        result = self.model(img, conf=conf)[0]
        output = {}
        if(len(result) > 0):
            result = result[0]
            # detection
            _cls = result.boxes.cls.cpu().numpy() # cls, (N, 1)
            conf = result.boxes.conf.cpu().numpy() # confidence score, (N, 1)
            
            # process the xywh
            xywh = result.boxes.xywh #.cpu().numpy() # box with xywh format, (N, 4)
            xywh[0][2] = box_size[0]
            xywh[0][3] = box_size[1]
            
            # get the box coords xyxy for cv.rectangle
            xyxy = xywh2xyxy(xywh.view(1, 4)).int()
            clip_coords(xyxy, result.masks.orig_shape)
            xyxy = self.pad_xyxy(xyxy, box_size)
            xyxy = xyxy.cpu().numpy() # convert tensor to numpy for cv2
            
            # segmentation
            masks = result.masks.masks.cpu().numpy() # masks, (N, H, W)
            masks = np.moveaxis(masks, 0, -1) # masks, (H, W, N)
            # rescale masks to original image
            masks = scale_image(masks.shape[:2], masks, result.masks.orig_shape)
            masks = np.moveaxis(masks, -1, 0) # masks, (N, H, W)
            
            output['xyxy'] = xyxy
            output['xywh'] = xywh
            output['masks'] = masks
            output['_cls'] = _cls
            output['conf'] = conf

        return output
    
    def pad_xyxy(self, xyxy, size):
        h, w = xyxy[0, 3] - xyxy[0, 1], xyxy[0, 2] - xyxy[0, 0]

        if (h, w) != size:
            xyxy[0, 2] += size[0] - w
            xyxy[0, 3] += size[0] - h

        assert (xyxy[0, 3] - xyxy[0, 1], xyxy[0, 2] - xyxy[0, 0]) == size

        return xyxy
    
    def draw_mask(self, image, mask, fill_values=(128, 128, 128), alpha=0.5):

        colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
        colored_mask = np.moveaxis(colored_mask, 0, -1)
        masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=fill_values)
        image_overlay = masked.filled()
        image_combined = cv2.addWeighted(image, 1-alpha, image_overlay, alpha, 0)
        return image_combined

    def draw_bbox(self, image, xyxy):
        image = cv2.rectangle(image, (int(xyxy[0][0]), int(xyxy[0][1])), (int(xyxy[0][2]), int(xyxy[0][3])), color=(128, 128, 128), thickness=3)
        return image
    
    def process_bbox(self, frame):
        output = self.yolo_predict(frame)
        if len(output) > 0:            
            frame = self.draw_bbox(frame, output['xyxy'])
        return frame

    def process_mask(self, frame):
        output = self.yolo_predict(frame)
        if len(output) > 0:
            frame = self.draw_mask(frame, output['masks'])
        return frame

    def process_6d(self, frame):
        output = self.yolo_predict(frame)
        if len(output) > 0:            
            center = (int(output['xywh'][0, 0]), int(output['xywh'][0, 1]))
            if len(self.previous_center) == 0: self.previous_center = center
            tx_gt_pose = center[0] - self.previous_center[0]
            ty_gt_pose = center[1] - self.previous_center[1]
            # get the predicted gt_pose through yolov8 bbox detection
            self.pose6d.gt_pose[0, 3] = self.pose6d.gt_pose[0, 3] + 0.01*tx_gt_pose
            self.pose6d.gt_pose[1, 3] = self.pose6d.gt_pose[1, 3] + 0.01*ty_gt_pose
            # get the right rgb color for both surfaces and image
            frame = self.pose6d.project_pose_cpu(frame[..., ::-1])[..., ::-1] # frame = self.pose6d.project_pose_gpu(frame[..., ::-1])[..., ::-1]
            self.previous_center = center
        return frame

def run_pose_prediction(model, ):
    cfg.model = root / "best.pt"
    cfg.source = source
    cfg.side = side
"""

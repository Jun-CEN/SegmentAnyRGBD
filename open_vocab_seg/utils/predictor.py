# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import numpy as np
import torch
import torchvision
import imageio
from tqdm import tqdm
import os
import cv2

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import look_at_view_transform

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data.detection_utils import read_image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import matplotlib.pyplot as plt
import matplotlib as mpl
from .pcd_rendering import unproject_pts_pt, get_coord_grids_pt, create_pcd_renderer


class OVSegPredictor(DefaultPredictor):
    def __init__(self, cfg):
        super().__init__(cfg)

    def __call__(self, original_image, class_names):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width, "class_names": class_names}
            predictions = self.model([inputs])[0]
            return predictions

class OVSegVisualizer(Visualizer):
    def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE, class_names=None):
        super().__init__(img_rgb, metadata, scale, instance_mode)
        self.class_names = class_names

    def draw_sem_seg(self, sem_seg, area_threshold=None, alpha=0.8):
        """
        Draw semantic segmentation predictions/labels.

        Args:
            sem_seg (Tensor or ndarray): the segmentation of shape (H, W).
                Each value is the integer label of the pixel.
            area_threshold (int): segments with less than `area_threshold` are not drawn.
            alpha (float): the larger it is, the more opaque the segmentations are.

        Returns:
            output (VisImage): image object with visualizations.
        """
        if isinstance(sem_seg, torch.Tensor):
            sem_seg = sem_seg.numpy()
        labels, areas = np.unique(sem_seg, return_counts=True)
        sorted_idxs = np.argsort(-areas).tolist()
        labels = labels[sorted_idxs]
        class_names = self.class_names if self.class_names is not None else self.metadata.stuff_classes

        for label in filter(lambda l: l < len(class_names), labels):
            try:
                mask_color = [x / 255 for x in self.metadata.stuff_colors[label]]
            except (AttributeError, IndexError):
                mask_color = None
            mask_color = np.random.random((1, 3)).tolist()[0]

            binary_mask = (sem_seg == label).astype(np.uint8)
            text = class_names[label]
            self.draw_binary_mask(
                binary_mask,
                color=mask_color,
                edge_color=(1.0, 1.0, 240.0 / 255),
                text=text,
                alpha=alpha,
                area_threshold=area_threshold,
            )
        return self.output

    def draw_sam_seg(self, masks, area_threshold=None, alpha=0.5):
        """
        Draw semantic segmentation predictions/labels.

        Args:
            sem_seg (Tensor or ndarray): the segmentation of shape (H, W).
                Each value is the integer label of the pixel.
            area_threshold (int): segments with less than `area_threshold` are not drawn.
            alpha (float): the larger it is, the more opaque the segmentations are.

        Returns:
            output (VisImage): image object with visualizations.
        """
        plt.figure()
        if len(masks) == 0:
            return
        sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
        class_names = self.class_names if self.class_names is not None else self.metadata.stuff_classes
        for ann in sorted_anns:
            m = ann['segmentation']
            mask_color = np.random.random((1, 3)).tolist()[0]
        
            self.draw_binary_mask(
                m,
                color=mask_color,
                edge_color=(1.0, 1.0, 240.0 / 255),
                text=class_names[ann['class']],
                alpha=alpha,
                area_threshold=area_threshold,
            )
        return self.output



class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )

        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            raise NotImplementedError
        else:
            self.predictor = OVSegPredictor(cfg)

    def run_on_image(self, image, class_names):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        predictions = self.predictor(image, class_names)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = OVSegVisualizer(image, self.metadata, instance_mode=self.instance_mode, class_names=class_names)
        # if "sem_seg" in predictions:
        #     r = predictions["sem_seg"]
        #     blank_area = (r[0] == 0)
        #     pred_mask = r.argmax(dim=0).to('cpu')
        #     pred_mask[blank_area] = 255
        #     pred_mask = np.array(pred_mask, dtype=np.int)

        #     vis_output = visualizer.draw_sem_seg(
        #         pred_mask
        #     )
        # else:
        #     raise NotImplementedError

        if "sem_seg" in predictions:
            r = predictions["sem_seg"]
            pred_mask = r.argmax(dim=0).to('cpu')
            pred_mask = np.array(pred_mask, dtype=np.int)

            vis_output = visualizer.draw_sem_seg(
                pred_mask
            )
        else:
            raise NotImplementedError
        
        return predictions, vis_output
    
    def run_on_image_sam(self, path, class_names, depth_map_path, rage_matrices_path):
        """
        Args:
            path (str): the path of the image
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        image = read_image(path, format="BGR")
        predictions = self.predictor(image, class_names)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer_rgb = OVSegVisualizer(image, self.metadata, instance_mode=self.instance_mode, class_names=class_names)
        visualizer_depth = OVSegVisualizer(image, self.metadata, instance_mode=self.instance_mode, class_names=class_names)
        visualizer_rgb_sam = OVSegVisualizer(image, self.metadata, instance_mode=self.instance_mode, class_names=class_names)
        visualizer_depth_sam = OVSegVisualizer(image, self.metadata, instance_mode=self.instance_mode, class_names=class_names)
        
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        mask_generator_2 = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=64,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.8,
            crop_n_layers=0,
            crop_n_points_downscale_factor=0,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
        )
        print('Using SAM to generate segments for the RGB image')
        masks_rgb = mask_generator_2.generate(image)

        print('Using SAM to generate segments for the Depth map')
        d, world_coord = self.project_2d_to_3d(depth_map_path, rage_matrices_path)
        d = (d - np.min(d)) / (np.max(d) - np.min(d))
        image_depth = mpl.colormaps['plasma'](d)*255
        plt.figure()
        plt.imshow(image_depth.astype(np.uint8))
        plt.axis('off')
        plt.savefig('outputs/Depth_rendered.png', bbox_inches='tight', pad_inches=0.0)
        masks_depth = mask_generator_2.generate(image_depth.astype(np.uint8)[:,:,:-1])

        if "sem_seg" in predictions:
            r = predictions["sem_seg"]
            pred_mask = r.argmax(dim=0).to('cpu')
            pred_mask = np.array(pred_mask, dtype=np.int)
            
            pred_mask_sam_rgb = pred_mask.copy()
            for mask in masks_rgb:
                cls_tmp, cls_num = np.unique(pred_mask_sam_rgb[mask['segmentation']], return_counts=True)
                pred_mask_sam_rgb[mask['segmentation']] = cls_tmp[np.argmax(cls_num)]
                mask['class'] = cls_tmp[np.argmax(cls_num)]

            vis_output_rgb = visualizer_rgb.draw_sem_seg(
                pred_mask_sam_rgb
            )
            # vis_output_rgb = visualizer_rgb.draw_sem_seg(
            #     pred_mask, alpha=1
            # )

            pred_mask_sam_depth = pred_mask.copy()
            for mask in masks_depth:
                cls_tmp, cls_num = np.unique(pred_mask_sam_depth[mask['segmentation']], return_counts=True)
                pred_mask_sam_depth[mask['segmentation']] = cls_tmp[np.argmax(cls_num)]
                mask['class'] = cls_tmp[np.argmax(cls_num)]

            vis_output_depth = visualizer_depth.draw_sem_seg(
                pred_mask_sam_depth
            )

            vis_output_rgb_sam = visualizer_rgb_sam.draw_sam_seg(masks_rgb)
            vis_output_depth_sam = visualizer_depth_sam.draw_sam_seg(masks_depth)

        else:
            raise NotImplementedError
        
        return predictions, vis_output_rgb, vis_output_depth, vis_output_rgb_sam, vis_output_depth_sam
    
    def project_2d_to_3d(self, depth_map_path, rage_matrices_path):

        H = 800
        W = 1280
        IMAGE_SIZE = (H, W)

        def pixels_to_ndcs(xx, yy, size=IMAGE_SIZE):
            s_y, s_x = size
            s_x -= 1  # so 1 is being mapped into (n-1)th pixel
            s_y -= 1  # so 1 is being mapped into (n-1)th pixel
            x = (2 / s_x) * xx - 1
            y = (-2 / s_y) * yy + 1
            return x, y

        rage_matrices = np.load(rage_matrices_path)


        # get the (ViewProj) matrix that transform points from the world coordinate to NDC
        # (points in world coordinate) @ VP = (points in NDC) 
        VP = rage_matrices['VP']
        VP_inverse = rage_matrices['VP_inv'] # NDC to world coordinate

        # get the (Proj) matrix that transform points from the camera coordinate to NDC
        # (points in camera coordinate) @ P = (points in NDC) 
        P = rage_matrices['P']
        P_inverse = rage_matrices['P_inv'] # NDC to camera coordinate
        # print(VP, VP_inverse, P, P_inverse)

        d = np.load(depth_map_path)
        d = d/6.0 - 4e-5 # convert to NDC coordinate

        px = np.arange(0, W)
        py = np.arange(0, H)
        px, py = np.meshgrid(px, py, sparse=False)
        px = px.reshape(-1)
        py = py.reshape(-1)

        ndcz = d[py, px] # get the depth in NDC
        ndcx, ndcy = pixels_to_ndcs(px, py)
        ndc_coord = np.stack([ndcx, ndcy, ndcz, np.ones_like(ndcz)], axis=1)

        camera_coord = ndc_coord @ P_inverse
        camera_coord = camera_coord/camera_coord[:,-1:]

        world_coord = ndc_coord @ VP_inverse
        world_coord = world_coord/world_coord[:,-1:]

        return d, world_coord

    def get_xyzrgb(self, rgb_path, depth_path, rage_matrices_path):

        H = 800
        W = 1280
        IMAGE_SIZE = (H, W)

        def pixels_to_ndcs(xx, yy, size=IMAGE_SIZE):
            s_y, s_x = size
            s_x -= 1  # so 1 is being mapped into (n-1)th pixel
            s_y -= 1  # so 1 is being mapped into (n-1)th pixel
            x = (2 / s_x) * xx - 1
            y = (-2 / s_y) * yy + 1
            return x, y
        
        rage_matrices = np.load(rage_matrices_path)


        # get the (ViewProj) matrix that transform points from the world coordinate to NDC
        # (points in world coordinate) @ VP = (points in NDC) 
        VP = rage_matrices['VP']
        VP_inverse = rage_matrices['VP_inv'] # NDC to world coordinate

        # get the (Proj) matrix that transform points from the camera coordinate to NDC
        # (points in camera coordinate) @ P = (points in NDC) 
        P = rage_matrices['P']
        P_inverse = rage_matrices['P_inv'] # NDC to camera coordinate
        # print(VP, VP_inverse, P, P_inverse)

        d = np.load(depth_path)
        d = d/6.0 - 4e-5 # convert to NDC coordinate

        px = np.arange(0, W)
        py = np.arange(0, H)
        px, py = np.meshgrid(px, py, sparse=False)
        px = px.reshape(-1)
        py = py.reshape(-1)

        ndcz = d[py, px] # get the depth in NDC
        ndcx, ndcy = pixels_to_ndcs(px, py)
        ndc_coord = np.stack([ndcx, ndcy, ndcz, np.ones_like(ndcz)], axis=1)

        camera_coord = ndc_coord @ P_inverse
        camera_coord = camera_coord/camera_coord[:,-1:]

        world_coord = ndc_coord @ VP_inverse
        world_coord = world_coord/world_coord[:,-1:]

        rgb = read_image(rgb_path, format="BGR")
        rgb = rgb[:, :, ::-1]
        rgb = rgb[py, px, :]

        xyzrgb = np.concatenate((world_coord[:,:-1], rgb), axis=1)

        return xyzrgb
    
    def render_3d_video(self, xyzrgb_path, depth_path):
        
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        xyzrgb = np.load(xyzrgb_path)
        depth = np.load(depth_path)
        depth = torch.tensor(depth).to(device)
        depth = 1 / depth
        
        H = 800
        W = 1280
        radius = 1.5 / min(H, W) * 2.0
        intrinsic = np.array([[max(H, W), 0, W // 2],
                              [0, max(H, W), H // 2],
                              [0, 0, 1]])

        intrinsic = torch.from_numpy(intrinsic).float()[None].to(device)
        coord = get_coord_grids_pt(H, W, device=device).float()[None]
        pts = unproject_pts_pt(intrinsic, coord.reshape(-1, 2), depth)
        pts[:, 0] = ((pts[:, 0] - pts[:, 0].min()) / (pts[:, 0].max() - pts[:, 0].min()) - 0.5) * 2
        pts[:, 1] = ((pts[:, 1] - pts[:, 1].min()) / (pts[:, 1].max() - pts[:, 1].min()) - 0.7) * 2
        pts[:, 2] = ((pts[:, 2] - pts[:, 2].min()) / (pts[:, 2].max() - pts[:, 2].min()) - 0.5) * 2
        
        num_frames = 45
        degrees = np.linspace(120, 220, num_frames)
        
        total = ['rgb_3d_sam', 'depth_3d_sam', 'rgb_3d_sam_mask', 'depth_3d_sam_mask']
        
        for j, name in enumerate(total):
            img = torch.from_numpy(xyzrgb[name][:, 3:] / 255.).to(device).float()
            pcd = Pointclouds(points=[pts], features=[img.squeeze().reshape(-1, 3)])
            frames = []
            for i in tqdm(range(num_frames)):
                R, t = look_at_view_transform(3., -10, degrees[i])
                renderer = create_pcd_renderer(H, W, intrinsic.squeeze()[:3, :3],
                                                           R=R, T=t,
                                                           radius=radius, device=device)
                result = renderer(pcd)
                result = result.permute(0, 3, 1, 2)
                frame = (255. * result.detach().cpu().squeeze().permute(1, 2, 0).numpy()).astype(np.uint8)
                frames.append(frame)

            video_out_file = '{}.gif'.format(name)
            imageio.mimwrite(os.path.join('outputs', video_out_file), frames, fps=25)
            
            video_out_file = '{}.mp4'.format(name)
            imageio.mimwrite(os.path.join('outputs', video_out_file), frames, fps=25, quality=8)
            
class VisualizationDemoIndoor(VisualizationDemo):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        super().__init__(cfg, instance_mode, parallel)

    def build_pcd(self, depth_mask,  coords, colors, masks, sem_map):
        group_ids = np.full(masks[0]["segmentation"].shape, -1, dtype=int)
        num_masks = len(masks)
        group_counter = 0
        for i in reversed(range(num_masks)):
            # print(masks[i]["predicted_iou"])
            group_ids[masks[i]["segmentation"]] = group_counter
            group_counter += 1
        group_ids = np.unique(group_ids[depth_mask], return_inverse=True)[1]
        return dict(coord=coords, color=colors, group=group_ids, sem_map=sem_map)


    def run_on_pcd_ui(self, rgb_path, depth_path, class_names):
        depth = depth_path
        color = rgb_path
        #semantic_map = join(rgb_path, scene_name, 'semantic_label', color_name[0:-4] + '.pth')

        depth_img = cv2.imread(depth, -1) # read 16bit grayscale image
        depth_mask = (depth_img != 0)
        color_image = cv2.imread(color)
        color_image = cv2.resize(color_image, (640, 480))
        predictions = self.predictor(color_image, class_names)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = color_image[:, :, ::-1]
        visualizer_rgb = OVSegVisualizer(image, self.metadata, instance_mode=self.instance_mode, class_names=class_names)
        visualizer_depth = OVSegVisualizer(image, self.metadata, instance_mode=self.instance_mode, class_names=class_names)
        visualizer_rgb_sam = OVSegVisualizer(image, self.metadata, instance_mode=self.instance_mode, class_names=class_names)
        visualizer_depth_sam = OVSegVisualizer(image, self.metadata, instance_mode=self.instance_mode, class_names=class_names)
        
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        mask_generator_2 = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=64,
            pred_iou_thresh=0.5,
            stability_score_thresh=0.8,
            crop_n_layers=0,
            crop_n_points_downscale_factor=0,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
        )
        print('Using SAM to generate segments for the RGB image')
        masks_rgb = mask_generator_2.generate(image)

        print('Using SAM to generate segments for the Depth map')
        d = np.full(depth_img.shape, 0, dtype=float)
        d[depth_mask] = (1 / (depth_img+1e-6))[depth_mask]
        colored_depth = (d - np.min(d)) / (np.max(d) - np.min(d))
        colored_depth = mpl.colormaps['inferno'](colored_depth)*255
        plt.figure()
        plt.imshow(colored_depth.astype(np.uint8)[:,:,:-1])
        plt.axis('off')
        plt.savefig('outputs/Depth_rendered.png')
        masks_depth = mask_generator_2.generate(colored_depth.astype(np.uint8)[:,:,:-1])

        if "sem_seg" in predictions:
            r = predictions["sem_seg"]
            pred_mask = r.argmax(dim=0).to('cpu')
            pred_mask = np.array(pred_mask, dtype=int)

            output2D = {}
            pred_mask_sam_depth = np.full(pred_mask.shape, -1)
            masks_depth = sorted(masks_depth, key=(lambda x: x['area']), reverse=False)
            for mask in masks_depth:
                to_paint = pred_mask_sam_depth == -1
                cls_tmp, cls_num = np.unique(pred_mask[mask['segmentation']], return_counts=True)
                #print(cls_tmp, cls_num)
                pred_mask_sam_depth[mask['segmentation'] & to_paint] = cls_tmp[np.argmax(cls_num)]
                #print(class_names[cls_tmp[np.argmax(cls_num)]])
                mask['class'] = cls_tmp[np.argmax(cls_num)]

            output2D['sem_seg_on_depth'] = visualizer_depth.draw_sem_seg(
                pred_mask_sam_depth
            )
            
            pred_mask_sam_rgb = pred_mask.copy()
            for mask in masks_rgb:
                cls_tmp, cls_num = np.unique(pred_mask[mask['segmentation']], return_counts=True)
                #print(mask['segmentation'].sum(), cls_tmp, cls_num)
                pred_mask_sam_rgb[mask['segmentation']] = cls_tmp[np.argmax(cls_num)]
                mask['class'] = cls_tmp[np.argmax(cls_num)]

            output2D['sem_seg_on_rgb'] = visualizer_rgb.draw_sem_seg(
                pred_mask_sam_rgb
            )

            output2D['sam_seg_on_rgb'] = visualizer_rgb_sam.draw_sam_seg(masks_rgb)
            output2D['sam_seg_on_depth'] = visualizer_depth_sam.draw_sam_seg(masks_depth)

        else:
            raise NotImplementedError
        
        color_image = np.reshape(color_image[depth_mask], [-1,3])
        #group_ids = group_ids[depth_mask]

        sem_map_color = pred_mask_sam_rgb[depth_mask]
        sem_map_depth = pred_mask_sam_depth[depth_mask]

        colors = np.zeros_like(color_image)
        colors[:,0] = color_image[:,2]
        colors[:,1] = color_image[:,1]
        colors[:,2] = color_image[:,0]
        
        depth_shift = 1000.0
        x,y = np.meshgrid(np.linspace(0,depth_img.shape[1]-1,depth_img.shape[1]), np.linspace(0,depth_img.shape[0]-1,depth_img.shape[0]))
        uv_depth = np.zeros((depth_img.shape[0], depth_img.shape[1], 3))
        uv_depth[:,:,0] = x
        uv_depth[:,:,1] = y
        uv_depth[:,:,2] = depth_img/depth_shift

        output3D = {}
        output3D['rgb_3d_sem'] = np.stack((uv_depth, output2D['sem_seg_on_rgb'].get_image()), axis=2).reshape((depth_img.shape[0], depth_img.shape[1], 6))
        output3D['depth_3d_sem'] = np.stack((uv_depth, output2D['sem_seg_on_rgb'].get_image()), axis=2).reshape((depth_img.shape[0], depth_img.shape[1], 6))
        output3D['rgb_3d_sam'] = np.stack((uv_depth, output2D['sam_seg_on_rgb'].get_image()), axis=2).reshape((depth_img.shape[0], depth_img.shape[1], 6))
        output3D['depth_3d_sam'] = np.stack((uv_depth, output2D['sam_seg_on_depth'].get_image()), axis=2).reshape((depth_img.shape[0], depth_img.shape[1], 6))
        
        return predictions, output2D, output3D
    
    def run_on_pcd(self, rgb_path, scene_name, color_name, class_names):
        intrinsic_path = os.path.join(rgb_path, scene_name, 'intrinsics', 'intrinsic_depth.txt')
        depth_intrinsic = np.loadtxt(intrinsic_path)

        pose = os.path.join(rgb_path, scene_name, 'pose', color_name[0:-4] + '.txt')
        depth = os.path.join(rgb_path, scene_name, 'depth', color_name[0:-4] + '.png')
        color = os.path.join(rgb_path, scene_name, 'color', color_name)
        #semantic_map = join(rgb_path, scene_name, 'semantic_label', color_name[0:-4] + '.pth')

        depth_img = cv2.imread(depth, -1) # read 16bit grayscale image
        depth_mask = (depth_img != 0)
        color_image = cv2.imread(color)
        color_image = cv2.resize(color_image, (640, 480))
        predictions = self.predictor(color_image, class_names)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = color_image[:, :, ::-1]
        visualizer_rgb = OVSegVisualizer(image, self.metadata, instance_mode=self.instance_mode, class_names=class_names)
        visualizer_depth = OVSegVisualizer(image, self.metadata, instance_mode=self.instance_mode, class_names=class_names)
        visualizer_rgb_sam = OVSegVisualizer(image, self.metadata, instance_mode=self.instance_mode, class_names=class_names)
        visualizer_depth_sam = OVSegVisualizer(image, self.metadata, instance_mode=self.instance_mode, class_names=class_names)
        
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        mask_generator_2 = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=64,
            pred_iou_thresh=0.5,
            stability_score_thresh=0.8,
            crop_n_layers=0,
            crop_n_points_downscale_factor=0,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
        )
        print('Using SAM to generate segments for the RGB image')
        masks_rgb = mask_generator_2.generate(image)

        print('Using SAM to generate segments for the Depth map')  
        d = np.full(depth_img.shape, 0, dtype=float)
        d[depth_mask] = (1 / (depth_img+1e-6))[depth_mask]
        colored_depth = (d - np.min(d)) / (np.max(d) - np.min(d))
        colored_depth = mpl.colormaps['inferno'](colored_depth)*255
        plt.figure()
        plt.imshow(colored_depth.astype(np.uint8)[:,:,:-1])
        plt.axis('off')
        plt.savefig('outputs/Depth_rendered.png')
        masks_depth = mask_generator_2.generate(colored_depth.astype(np.uint8)[:,:,:-1])

        if "sem_seg" in predictions:
            r = predictions["sem_seg"]
            pred_mask = r.argmax(dim=0).to('cpu')
            pred_mask = np.array(pred_mask, dtype=int)

            output2D = {}
            pred_mask_sam_depth = np.full(pred_mask.shape, -1)
            masks_depth = sorted(masks_depth, key=(lambda x: x['area']), reverse=False)
            for mask in masks_depth:
                to_paint = pred_mask_sam_depth == -1
                cls_tmp, cls_num = np.unique(pred_mask[mask['segmentation']], return_counts=True)
                #print(cls_tmp, cls_num)
                pred_mask_sam_depth[mask['segmentation'] & to_paint] = cls_tmp[np.argmax(cls_num)]
                #print(class_names[cls_tmp[np.argmax(cls_num)]])
                mask['class'] = cls_tmp[np.argmax(cls_num)]

            output2D['sem_seg_on_depth'] = visualizer_depth.draw_sem_seg(
                pred_mask_sam_depth
            )
            
            pred_mask_sam_rgb = pred_mask.copy()
            for mask in masks_rgb:
                cls_tmp, cls_num = np.unique(pred_mask[mask['segmentation']], return_counts=True)
                #print(mask['segmentation'].sum(), cls_tmp, cls_num)
                pred_mask_sam_rgb[mask['segmentation']] = cls_tmp[np.argmax(cls_num)]
                mask['class'] = cls_tmp[np.argmax(cls_num)]

            output2D['sem_seg_on_rgb'] = visualizer_rgb.draw_sem_seg(
                pred_mask_sam_rgb
            )

            output2D['sam_seg_on_rgb'] = visualizer_rgb_sam.draw_sam_seg(masks_rgb)
            output2D['sam_seg_on_depth'] = visualizer_depth_sam.draw_sam_seg(masks_depth)

        else:
            raise NotImplementedError
        
        color_image = np.reshape(color_image[depth_mask], [-1,3])
        #group_ids = group_ids[depth_mask]

        sem_map_color = pred_mask_sam_rgb[depth_mask]
        sem_map_depth = pred_mask_sam_depth[depth_mask]

        colors = np.zeros_like(color_image)
        colors[:,0] = color_image[:,2]
        colors[:,1] = color_image[:,1]
        colors[:,2] = color_image[:,0]

        pose = np.loadtxt(pose)
        
        depth_shift = 1000.0
        x,y = np.meshgrid(np.linspace(0,depth_img.shape[1]-1,depth_img.shape[1]), np.linspace(0,depth_img.shape[0]-1,depth_img.shape[0]))
        uv_depth = np.zeros((depth_img.shape[0], depth_img.shape[1], 3))
        uv_depth[:,:,0] = x
        uv_depth[:,:,1] = y
        uv_depth[:,:,2] = depth_img/depth_shift

        output3D = {}
        output3D['rgb_3d_sem'] = np.stack((uv_depth, output2D['sem_seg_on_rgb'].get_image()), axis=2).reshape((depth_img.shape[0], depth_img.shape[1], 6))
        output3D['depth_3d_sem'] = np.stack((uv_depth, output2D['sem_seg_on_rgb'].get_image()), axis=2).reshape((depth_img.shape[0], depth_img.shape[1], 6))
        output3D['rgb_3d_sam'] = np.stack((uv_depth, output2D['sam_seg_on_rgb'].get_image()), axis=2).reshape((depth_img.shape[0], depth_img.shape[1], 6))
        output3D['depth_3d_sam'] = np.stack((uv_depth, output2D['sam_seg_on_depth'].get_image()), axis=2).reshape((depth_img.shape[0], depth_img.shape[1], 6))

        uv_depth = np.reshape(uv_depth, [-1,3])
        uv_depth = uv_depth[np.where(uv_depth[:,2]!=0),:].squeeze()
        
        intrinsic_inv = np.linalg.inv(depth_intrinsic)
        fx = depth_intrinsic[0,0]
        fy = depth_intrinsic[1,1]
        cx = depth_intrinsic[0,2]
        cy = depth_intrinsic[1,2]
        bx = depth_intrinsic[0,3]
        by = depth_intrinsic[1,3]
        n = uv_depth.shape[0]
        points = np.ones((n,4))
        X = (uv_depth[:,0]-cx)*uv_depth[:,2]/fx + bx
        Y = (uv_depth[:,1]-cy)*uv_depth[:,2]/fy + by
        points[:,0] = X
        points[:,1] = Y
        points[:,2] = uv_depth[:,2]
        points_world = np.dot(points, np.transpose(pose))
        
        output3D['pcd_color'] = self.build_pcd(depth_mask, coords=points_world[:,:3], colors=colors, masks=masks_rgb, sem_map=sem_map_color)
        output3D['pcd_depth'] = self.build_pcd(depth_mask, coords=points_world[:,:3], colors=colors, masks=masks_depth, sem_map=sem_map_depth)
        
        return predictions, output2D, output3D
    
    
    def merge_pcd(self, pcd_list, data_path, save_path, scene_path, voxel_size, th):
        while len(pcd_list) != 1:
            print(len(pcd_list), flush=True)
            new_pcd_list = []
            for indice in pairwise_indices(len(pcd_list)):
                # print(indice)
                pcd_frame = cal_2_scenes(pcd_list, indice, voxel_size=voxel_size, voxelize=voxelize)
                if pcd_frame is not None:
                    new_pcd_list.append(pcd_frame)
            pcd_list = new_pcd_list
        seg_dict = pcd_list[0]
        seg_dict["group"] = num_to_natural(remove_small_group(seg_dict["group"], th))

        data_dict = torch.load(scene_path)
        scene_coord = torch.tensor(data_dict["coord"]).cuda().contiguous()
        new_offset = torch.tensor(scene_coord.shape[0]).cuda()
        gen_coord = torch.tensor(seg_dict["coord"]).cuda().contiguous().float()
        offset = torch.tensor(gen_coord.shape[0]).cuda()
        gen_group = seg_dict["group"]
        gen_sem = seg_dict['sem_map']
        indices, dis = pointops.knn_query(1, gen_coord, offset, scene_coord, new_offset)
        indices = indices.cpu().numpy()
        sem_map = gen_sem[indices.reshape(-1)].astype(np.int16)
        group = gen_group[indices.reshape(-1)].astype(np.int16)
        mask_dis = dis.reshape(-1).cpu().numpy() > 0.6
        group[mask_dis] = -1
        sem_map[mask_dis] = -1
        group = group.astype(np.int16)
        sem_map = sem_map.astype(np.int16)
        torch.save((sem_map, num_to_natural(group)), os.path.join(save_path, scene_name + ".pth"))
    
    def render_3d_video(self, xyzrgb_path):
        xyzrgb = np.load(xyzrgb_path)
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        depth = xyzrgb['rgb_3d_sam'][:, :, 2]
        depth = torch.tensor(depth).to(device).float()

        num_frames = [60, 60, 60, 90]

        h = 480
        w = 640

        intrinsic = np.array([[max(h, w), 0, w // 2],
                            [0, max(h, w), h // 2],
                            [0, 0, 1]])
        intrinsic = torch.from_numpy(intrinsic).float()[None].to(device)

        coord = get_coord_grids_pt(h, w, device=device).float()[None]
        pts = unproject_pts_pt(intrinsic, coord.reshape(-1, 2), depth)
        pts[:, 0] = ((pts[:, 0] - pts[:, 0].min()) / (pts[:, 0].max() - pts[:, 0].min()) - 0.5) * 2
        pts[:, 1] = ((pts[:, 1] - pts[:, 1].min()) / (pts[:, 1].max() - pts[:, 1].min()) - 0.5) * 2
        # pts[:, 1] = ((pts[:, 1] - pts[:, 1].min()) / (pts[:, 1].max() - pts[:, 1].min()) - 0.7) * 2
        pts[:, 2] = ((pts[:, 2] - pts[:, 2].min()) / (pts[:, 2].max() - pts[:, 2].min()) - 0.5) * 2

        radius = 1.5 / min(h, w) * 2.0


        total = ['rgb_3d_sam', 'depth_3d_sam', 'rgb_3d_sam_mask', 'depth_3d_sam_mask']
        num_frames = 45
        degrees = np.linspace(120, 220, num_frames)
        for j, name in enumerate(total):
            img = torch.from_numpy(xyzrgb[name][:, :, 3:] / 255.).to(device).float()
            pcd = Pointclouds(points=[pts], features=[img.squeeze().reshape(-1, 3)])
            time_steps = np.linspace(0, 1, num_frames)
            frames = []
            for i, t_step in tqdm(enumerate(time_steps), total=len(time_steps)):
                R, t = look_at_view_transform(3., -10, degrees[i])
                renderer = create_pcd_renderer(h, w, intrinsic.squeeze()[:3, :3],
                                            R=R, T=t,
                                            radius=radius, device=device)

                result = renderer(pcd)
                result = result.permute(0, 3, 1, 2)
                frame = (255. * result.detach().cpu().squeeze().permute(1, 2, 0).numpy()).astype(np.uint8)
                frames.append(frame)

            video_out_file = '{}.gif'.format(name)
            imageio.mimwrite(os.path.join('outputs', video_out_file), frames, fps=25)
            
            video_out_file = '{}.mp4'.format(name)
            imageio.mimwrite(os.path.join('outputs', video_out_file), frames, fps=25, quality=8)

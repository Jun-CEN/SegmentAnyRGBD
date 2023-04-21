# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import numpy as np
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data.detection_utils import read_image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import matplotlib.pyplot as plt
import matplotlib as mpl


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

    def draw_sam_seg(self, masks, area_threshold=None, alpha=0.8):
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
    
    def run_on_image_sam(self, path, class_names):
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
        d, world_coord = self.project_2d_to_3d(path)
        d = (d - np.min(d)) / (np.max(d) - np.min(d))
        image_depth = mpl.colormaps['viridis'](d)*255
        plt.figure()
        plt.imshow(image_depth.astype(np.uint8))
        plt.axis('off')
        plt.savefig('Depth_rendered.png')
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
    
    def project_2d_to_3d(self, image_path):

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
        dataset_root = image_path[:-18]
        frameId = image_path[-10:-4]
        rage_matrices = np.load(dataset_root+'/rage_matrices/{}.npz'.format(frameId))


        # get the (ViewProj) matrix that transform points from the world coordinate to NDC
        # (points in world coordinate) @ VP = (points in NDC) 
        VP = rage_matrices['VP']
        VP_inverse = rage_matrices['VP_inv'] # NDC to world coordinate

        # get the (Proj) matrix that transform points from the camera coordinate to NDC
        # (points in camera coordinate) @ P = (points in NDC) 
        P = rage_matrices['P']
        P_inverse = rage_matrices['P_inv'] # NDC to camera coordinate
        # print(VP, VP_inverse, P, P_inverse)

        d = np.load(dataset_root+'/depth/{}.npy'.format(frameId))
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

    def get_xyzrgb(self, rgb_path, image_path):

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
        dataset_root = image_path[:-18]
        frameId = image_path[-10:-4]
        rage_matrices = np.load(dataset_root+'/rage_matrices/{}.npz'.format(frameId))


        # get the (ViewProj) matrix that transform points from the world coordinate to NDC
        # (points in world coordinate) @ VP = (points in NDC) 
        VP = rage_matrices['VP']
        VP_inverse = rage_matrices['VP_inv'] # NDC to world coordinate

        # get the (Proj) matrix that transform points from the camera coordinate to NDC
        # (points in camera coordinate) @ P = (points in NDC) 
        P = rage_matrices['P']
        P_inverse = rage_matrices['P_inv'] # NDC to camera coordinate
        # print(VP, VP_inverse, P, P_inverse)

        d = np.load(dataset_root+'/depth/{}.npy'.format(frameId))
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
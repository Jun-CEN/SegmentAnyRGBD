# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
from matplotlib import pyplot as plt
import tqdm
import numpy as np
import matplotlib as mpl
from tools.util import *

from detectron2.config import get_cfg

from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from open_vocab_seg import add_ovseg_config

from open_vocab_seg.utils import VisualizationDemo, OVSegVisualizer
from sam3d import *

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
        print(depth_img.shape, depth_img.max())
        #colored_depth = depth_img / 2**16
        
        d = np.full(depth_img.shape, 0, dtype=float)
        d[depth_mask] = (1 / (depth_img+1e-6))[depth_mask]
        print(depth_mask.sum(), d.max())
        colored_depth = (d - np.min(d)) / (np.max(d) - np.min(d))
        colored_depth = mpl.colormaps['inferno'](colored_depth)*255
        plt.figure()
        plt.imshow(colored_depth.astype(np.uint8)[:,:,:-1])
        plt.axis('off')
        plt.savefig('Depth_rendered.png')
        masks_depth = mask_generator_2.generate(colored_depth.astype(np.uint8)[:,:,:-1])

        if "sem_seg" in predictions:
            r = predictions["sem_seg"]
            pred_mask = r.argmax(dim=0).to('cpu')
            pred_mask = np.array(pred_mask, dtype=np.int)

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
        output3D['rgb_3d_sem'] = np.stack((uv_depth, output2D['sem_seg_on_rgb'].get_image()), axis=2)
        output3D['depth_3d_sem'] = np.stack((uv_depth, output2D['sem_seg_on_rgb'].get_image()), axis=2)
        output3D['rgb_3d_sam'] = np.stack((uv_depth, output2D['sam_seg_on_rgb'].get_image()), axis=2)
        output3D['depth_3d_sam'] = np.stack((uv_depth, output2D['sam_seg_on_depth'].get_image()), axis=2)

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
        
# constants
WINDOW_NAME = "Open vocabulary segmentation"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for open vocabulary segmentation")
    parser.add_argument(
        "--config-file",
        default="configs/ovseg_swinB_vitL_demo.yaml",
        metavar="FILE",
        help="path to config file",
    )
    """ parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        help="A list of user-defined class_names"
    ) """
    """ parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    ) """
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    parser.add_argument('--rgb_path', type=str, help='the path of rgb data')
    parser.add_argument('--data_path', type=str, default='', help='the path of pointcload data')
    parser.add_argument('--save_path', type=str, help='Where to save the pcd results')
    parser.add_argument('--save_2dmask_path', type=str, default='', help='Where to save 2D segmentation result from SAM')
    parser.add_argument('--sam_checkpoint_path', type=str, default='', help='the path of checkpoint for SAM')
    parser.add_argument('--scannetv2_train_path', type=str, default='datasets/scannet_preprocess/meta_data/scannetv2_train.txt', help='the path of scannetv2_train.txt')
    parser.add_argument('--scannetv2_val_path', type=str, default='datasets/scannet_preprocess/meta_data/scannetv2_val.txt', help='the path of scannetv2_val.txt')
    parser.add_argument('--img_size', default=[640,480])
    parser.add_argument('--voxel_size', default=0.05)
    parser.add_argument('--th', default=50, help='threshold of ignoring small groups to avoid noise pixel')
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemoIndoor(cfg)
    from datasets.scannet_preprocess.meta_data.scannet200_constants import CLASS_LABELS_200
    class_names = list(CLASS_LABELS_200)
    if True:
        if True:
            """ args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found" """
            voxelize = Voxelize(voxel_size=args.voxel_size, mode="train", keys=("coord", "color", "group", "sem_map"))
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            scene_names = sorted(os.listdir(args.rgb_path))
            for scene_name in scene_names:
                # use PIL, to be consistent with evaluation
                if os.path.exists(os.path.join(args.save_path, scene_name + ".pth")):
                    continue
                color_names = sorted(os.listdir(os.path.join(args.rgb_path, scene_name, 'color')), key=lambda a: int(os.path.basename(a).split('.')[0]), reverse=True)
                pcd_list = []
                for i, color_name in enumerate(color_names):
                    if i>5: break
                    path = os.path.join(args.rgb_path, scene_name, 'color', color_name)
                    start_time = time.time()
                    print(color_name, flush=True)
                    predictions, output2D, output3D = demo.run_on_pcd(args.rgb_path, scene_name, color_name, class_names)
                    pcd_dict = output3D['pcd_depth']
                    if len(pcd_dict["coord"]) == 0:
                        continue
                    pcd_dict = voxelize(pcd_dict)
                    pcd_list.append(pcd_dict)
                    
                    logger.info(
                        "{}: {} in {:.2f}s".format(
                            path,
                            "detected {} instances".format(len(predictions["instances"]))
                            if "instances" in predictions
                            else "finished",
                            time.time() - start_time,
                        )
                    )

                    if args.save_path:
                        if os.path.isdir(args.save_path):
                            assert os.path.isdir(args.save_path), args.save_path
                            out_filename = os.path.join(args.save_path, os.path.basename(path))
                        else:
                            assert len(args.input) == 1, "Please specify a directory with args.output"
                            out_filename = args.save_path
                        path = './'
                        output2D['sem_seg_on_rgb'].save('RGB_Semantic_SAM.png')
                        output2D['sem_seg_on_depth'].save('Depth_Semantic_SAM.png')
                        output2D['sam_seg_on_rgb'].save('RGB_Semantic_SAM_Mask.png')
                        output2D['sam_seg_on_depth'].save('Depth_Semantic_SAM_Mask.png')
                        """ rgb_3d_sam = demo.get_xyzrgb('RGB_Semantic_SAM.png', path)
                        depth_3d_sam = demo.get_xyzrgb('Depth_Semantic_SAM.png', path)
                        rgb_3d_sam_mask = demo.get_xyzrgb('RGB_Semantic_SAM_Mask.png', path)
                        depth_3d_sam_mask = demo.get_xyzrgb('Depth_Semantic_SAM_Mask.png', path) """
                        rgb_3d_sem = output3D['rgb_3d_sem']
                        depth_3d_sem = output3D['depth_3d_sem']
                        rgb_3d_sam = output3D['rgb_3d_sam']
                        depth_3d_sam = output3D['depth_3d_sam']
                        
                        np.savez('xyzrgb.npz', rgb_3d_sam = rgb_3d_sem, depth_3d_sam = depth_3d_sem, rgb_3d_sam_mask = rgb_3d_sam, depth_3d_sam_mask = depth_3d_sam)
                    else:
                        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                        cv2.imshow(WINDOW_NAME, output2D['sem_seg_on_rgb'].get_image()[:, :, ::-1])
                        if cv2.waitKey(0) == 27:
                            break  # esc to quit
                
                with open(args.scannetv2_train_path) as train_file:
                    train_scenes = train_file.read().splitlines()
                with open(args.scannetv2_val_path) as val_file:
                    val_scenes = val_file.read().splitlines()

                if scene_name in train_scenes:
                    scene_path = os.path.join(args.data_path, "train", scene_name + ".pth")
                elif scene_name in val_scenes:
                    scene_path = os.path.join(args.data_path, "val", scene_name + ".pth")

                demo.merge_pcd(pcd_list, args.data_path, args.save_path, scene_path, args.voxel_size, args.th)
    else:
        raise NotImplementedError
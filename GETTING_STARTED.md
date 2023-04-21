## Getting started with OVSeg


### Try demo

We release our largest model (Swin-Base + CLIP-ViT-L/14) [ovseg_swinbase_vitL14_ft_mpt.pth](https://drive.google.com/file/d/1cn-ohxgXDrDfkzC1QdO-fi8IjbjXmgKy/view?usp=sharing) (md5: <tt>526080</tt>).

- Test on sample image
  ```bash
  python demo.py --config-file configs/ovseg_swinB_vitL_demo.yaml --class-names 'Oculus' 'Ukulele'  --input ./resources/demo_samples/sample_03.jpeg --output ./pred --opts MODEL.WEIGHTS #PATH_of_ovseg_swinbase_vitL14_ft_mpt.pth 
  ```

### Evaluation with pre-trained weights

We release our largest model (Swin-Base + CLIP-ViT-L/14) [ovseg_swinbase_vitL14_ft_mpt.pth](https://drive.google.com/file/d/1cn-ohxgXDrDfkzC1QdO-fi8IjbjXmgKy/view?usp=sharing) (md5: <tt>526080</tt>).

- Test on ADE20K-150 and ADE-847
  ```bash
  python train_net.py --num-gpu 8 --eval-only --config-file configs/ovseg_swinB_vitL_bs32_120k.yaml MODEL.WEIGHTS #PATH_of_ovseg_swinbase_vitL14_ft_mpt.pth DATASETS.TEST \(\"ade20k_sem_seg_val\",\"ade20k_full_sem_seg_val\"\) 
  ```

- Test on PascalContext-59 and PascalContext-459
  ```bash
  python train_net.py --num-gpu 8 --eval-only --config-file configs/ovseg_swinB_vitL_bs32_120k.yaml MODEL.WEIGHTS #PATH_of_ovseg_swinbase_vitL14_ft_mpt.pth MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT 0.6  DATASETS.TEST \(\"pascal_context_59_sem_seg_val\",\"pascal_context_459_sem_seg_val\",\)
  ```

- Test on PascalVOC-20
  ```bash
  python train_net.py --num-gpu 8 --eval-only --config-file configs/ovseg_swinB_vitL_bs32_120k.yaml MODEL.WEIGHTS #PATH_of_ovseg_swinbase_vitL14_ft_mpt.pth MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT 0.45  DATASETS.TEST \(\"pascalvoc20_sem_seg_val\",\)
  ```
  
#### Performance benchmark

| method                             | backbone | training dataset | A-847 | PC-459 | A-150 | PC-59 | PAS-20 |
|------------------------------------|----------|------------------|:-----:|:------:|:-----:|:-----:|:------:|
| Open-vocabulary generalist models. |          |                  |       |        |       |       |        |
| SPNet                              | R-101    | PASCAL-15        |   -   |    -   |   -   |  24.3 |  18.3  |
| ZS3Net                             | R-101    | PASCAL-15        |   -   |    -   |   -   |  19.4 |  38.3  |
| LSeg                               | R-101    | PASCAL-15        |   -   |    -   |   -   |   -   |  47.4  |
| LSeg+                              | R-101    | COCO Panoptic    |  2.5  |   5.2  |  13.0 |  36.0 |  59.0  |
| SimBaseline                        | R-101c   | COCO-Stuff-156   |   -   |    -   |  15.3 |   -   |  74.5  |
| ZegFormer                          | R-50     | COCO-Stuff-156   |   -   |    -   |  16.4 |   -   |  80.7  |
| OpenSeg                            | R-101    | COCO Panoptic    |  4.0  |   6.5  |  15.3 |  36.9 |  60.0  |
| OVSeg (Ours)                       | R-101c   | COCO-Stuff-171   |  7.1  |  11.0  |  24.8 |  53.3 |  92.6  |
| LSeg+                              | Eff-B7   | COCO Panoptic    |  3.8  |   7.8  |  18.0 |  46.5 |    -   |
| OpenSeg                            | Eff-B7   | COCO Panoptic    |  6.3  |   9.0  |  21.1 |  42.1 |    -   |
| OVSeg (Ours)                       | Swin-B   | COCO-Stuff-171   |  9.0  |  12.4  |  29.6 |  55.7 |  94.5  |
| Supervised specialist models.      |          |                  |       |        |       |       |        |
| FCN                                | FCN-8s   | Same as test     |   -   |    -   |  29.4 |  37.8 |    -   |
| Deeplab                            | R-101    | Same as test     |   -   |    -   |   -   |  45.7 |  77.7  |
| SelfTrain                          | Eff-L2   | Same as test     |   -   |    -   |   -   |   -   |  90.0  |

#### Ablation study

- Mask prompt tuning can bring significant improvement without changing CLIP weights (Table 3 in [paper](https://arxiv.org/pdf/2210.04150.pdf))

Download the checkpoint with mpt only [ovseg_swinbase_vitL14_mpt_only.pt](https://drive.google.com/file/d/1LJGWFjHw76OGDNy9r9KQIaACfIm9KMhQ/view?usp=sharing) (md5: <tt>2dd495</tt>).

  ```bash
  python train_net.py --num-gpu 8 --eval-only --config-file configs/ovseg_swinB_vitL_bs32_120k.yaml MODEL.WEIGHTS #PATH_of_ovseg_swinbase_vitL14_mpt_only.pt DATASETS.TEST \(\"ade20k_sem_seg_val\",\"ade20k_full_sem_seg_val\"\) 
  ```
  
- Mask prompt tuning can improve over fully finetuned model (Table 3 in [paper](https://arxiv.org/pdf/2210.04150.pdf))

With the same [ovseg_swinbase_vitL14_ft_mpt.pth](https://drive.google.com/file/d/1cn-ohxgXDrDfkzC1QdO-fi8IjbjXmgKy/view?usp=sharing) checkpoint, set `MASK_PROMPT_FWD` as `False` 

  ```bash
  python train_net.py --num-gpu 8 --eval-only --config-file configs/ovseg_swinB_vitL_bs32_120k.yaml MODEL.CLIP_ADAPTER.MASK_PROMPT_FWD False MODEL.WEIGHTS #PATH_of_ovseg_swinbase_vitL14_ft_mpt.pth DATASETS.TEST \(\"ade20k_sem_seg_val\",\"ade20k_full_sem_seg_val\"\) 
  ```

- The effects of class prediction ensemble (Table 6 in [paper](https://arxiv.org/pdf/2210.04150.pdf))

With the same [ovseg_swinbase_vitL14_ft_mpt.pth](https://drive.google.com/file/d/1cn-ohxgXDrDfkzC1QdO-fi8IjbjXmgKy/view?usp=sharing) checkpoint, set `CLIP_ENSEMBLE` as `False`.

  ```bash
  python train_net.py --num-gpu 8 --eval-only --config-file configs/ovseg_swinB_vitL_bs32_120k.yaml MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE False MODEL.WEIGHTS #PATH_of_ovseg_swinbase_vitL14_ft_mpt.pth DATASETS.TEST \(\"ade20k_sem_seg_val\",\"ade20k_full_sem_seg_val\"\) 
  ```

### Training Segmentation model

  Our model is trained on COCO-Stuff
  
- Training baseline w/ original CLIP
  ```
  python train_net.py --num-gpu 8 --config-file configs/ovseg_swinB_vitL_bs32_120k.yaml MODEL.CLIP_ADAPTER.MASK_PROMPT_FWD False
  ```

To reproduce our final results, you may want to use the our mask-adapted CLIP

- Training ovseg w/ mask-adapted CLIP
  ```
  python train_net.py --num-gpu 8 --config-file configs/ovseg_swinB_vitL_bs32_120k.yaml MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME #PATH_TO_MASKADAPTED_CLIP
  ```
  
CAUTION: The final results is sensitive to the ensemble (appendix A.5 in [paper](https://arxiv.org/pdf/2210.04150.pdf)). Thus, you may want to use the ```tools/search_thr_ensemble_w.sh``` to find the best ensemble hyper-parameters.

### Fine-tuning CLIP with collected mask-category pairs

We are still working on this part, stay tuned!
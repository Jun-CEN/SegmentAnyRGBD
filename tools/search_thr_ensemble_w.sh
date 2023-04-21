or MASK_THR in 0.35 0.4 0.45
o
   for ENSEMBLE_WEIGHT in 0.6 0.65 0.7 0.75 0.8
   do
       python train_net.py --num-gpu 8 --eval-only --config-file configs/ovseg_swinB_vitL_bs32_120k.yaml \
       MODEL.WEIGHTS #PATH_of_ovseg_swinbase_vitL14_ft_mpt.pth DATASETS.TEST \(\"ade20k_sem_seg_val\"\) \
       MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT $ENSEMBLE_WEIGHT MODEL.CLIP_ADAPTER.MASK_THR $MASK_THR
   done
one



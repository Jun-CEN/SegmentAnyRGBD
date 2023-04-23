import cv2

Depth_Semantic_SAM_Mask_gif = cv2.VideoCapture('outputs/depth_3d_sam_mask.mp4')

while(Depth_Semantic_SAM_Mask_gif .isOpened()):  
    ret, frame = Depth_Semantic_SAM_Mask_gif.read()
    print(ret, frame.shape)
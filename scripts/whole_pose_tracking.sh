python3 tools/track_with_whole_pose.py \
    tracking/mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth \
    --device cuda:0 \
    --bbox-thr 0.4 \
    --video-path /home/ubuntu/Desktop/data_process/data/output2.avi \
    --out-video-root vis_results/detection_whole_2
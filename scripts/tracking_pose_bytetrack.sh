python3 tools/tracking_bytetrack.py video \
--path /home/ubuntu/vti_prj/ai-retail/data/test_1.mp4  \
-f tracking/ByteTrack/exps/example/mot/yolox_tiny_mix_det \
-c tracking/ByteTrack/pretrained/bytetrack_tiny_mot17.pth.tar \
--fp16 \
--fuse \
--save_result
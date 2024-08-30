# Example script to run SplatFields4D on the Owlii dataset
# Download the dataset via `gdown 1OdqwXKmvnpxFI4LC8ckI0eV6r9EZ64ZV && unzip DATA_OWLII`
DATASET_ROOT=./DATA_OWLII
SCENE=dancer # exercise model basketball
N_VIEWS=8 # 10 8 6 4

python train.py -s $DATASET_ROOT/$SCENE --white_background --eval --load_time_step 100  -m ./output_rep/Owlii/${N_VIEWS}views/$SCENE/SplatFields4D --flow_model offset --all_training --train_cam_names cam_train_0 cam_train_1 cam_train_2 cam_train_3 cam_train_4 cam_train_5 cam_train_6 cam_train_7 cam_train_8 cam_train_9 --pts_samples hull --iterations 200000 --encoder_type VarTriPlaneEncoder --num_pts 100000 --num_views 5 --composition_rank 40
python render.py -s $DATASET_ROOT/$SCENE --white_background --eval --load_time_step 100  -m ./output_rep/Owlii/${N_VIEWS}views/$SCENE/SplatFields4D --flow_model offset --all_training --train_cam_names cam_train_0 cam_train_1 cam_train_2 cam_train_3 cam_train_4 cam_train_5 cam_train_6 cam_train_7 cam_train_8 cam_train_9 --pts_samples hull --iterations 200000 --encoder_type VarTriPlaneEncoder --num_pts 100000 --num_views 5 --composition_rank 40

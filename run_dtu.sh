# Example script to run 3DGS and SplatFields3D on the DTU dataset

# RUN vanilla 3DGS on DTU dataset
# set the path to the DTU dataset
DATASET_ROOT=/media/STORAGE_4TB/NeRF_datasets/DTU/2DGS_data/dtu/DTU
SCENE=scan114 # scan105  scan106  scan110  scan114  scan118  scan122  scan24  scan37  scan40  scan55  scan63  scan65  scan69  scan83  scan97
N_VIEWS=3
python train.py -s $DATASET_ROOT/$SCENE -m ./output_rep/dtu/$SCENE/3views/3DGS --white_background --lambda_mask 0.1 -r 2 --is_static --n_views $N_VIEWS --iterations 30000
python render.py -s $DATASET_ROOT/$SCENE -m ./output_rep/dtu/$SCENE/3views/3DGS --white_background --lambda_mask 0.1 -r 2 --is_static --n_views $N_VIEWS --iterations 30000

# RUN SplatFields3D on DTU dataset
python train.py -s $DATASET_ROOT/$SCENE -m ./output_rep/dtu/$SCENE/3views/SplatFields3D --pc_path ./output_rep/dtu/$SCENE/3views/3DGS/point_cloud/iteration_1000/point_cloud.ply --deform_weight 0 --white_background --lambda_mask 0.1 --n_views $N_VIEWS --lambda_norm 0.01 --encoder_type VarTriPlaneEncoder --W 128 --iterations 30000 --max_num_pts 300000 -r 2 --load_time_step 0 --composition_rank 0
python render.py -s $DATASET_ROOT/$SCENE -m ./output_rep/dtu/$SCENE/3views/SplatFields3D --pc_path ./output_rep/dtu/$SCENE/3views/3DGS/point_cloud/iteration_1000/point_cloud.ply --deform_weight 0 --white_background --lambda_mask 0.1 --n_views $N_VIEWS --lambda_norm 0.01 --encoder_type VarTriPlaneEncoder --W 128 --iterations 30000 --max_num_pts 300000 -r 2 --load_time_step 0 --composition_rank 0

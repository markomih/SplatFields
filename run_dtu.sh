# ------------------------------------------------------------------------------

# RUN vanilla 3DGS on DTU dataset
python train.py -s /media/STORAGE_4TB/NeRF_datasets/DTU/2DGS_data/dtu/DTU/scan110 -m ./output_rep/dtu/scan110/3views/3DGS --white_background --lambda_mask 0.1 -r 2 --is_static --n_views 3 --iterations 30000
python render.py -s /media/STORAGE_4TB/NeRF_datasets/DTU/2DGS_data/dtu/DTU/scan110 -m ./output_rep/dtu/scan110/3views/3DGS --white_background --lambda_mask 0.1 -r 2 --is_static --n_views 3 --iterations 30000

# RUN SplatFields3D on DTU dataset
python train.py -s /media/STORAGE_4TB/NeRF_datasets/DTU/2DGS_data/dtu/DTU/scan110 -m ./output_rep/dtu/scan110/3views/SplatFields3D --pc_path ./output_rep/dtu/scan110/3views/3DGS/point_cloud/iteration_7000/point_cloud.ply --deform_weight 0 --white_background --lambda_mask 0.1 --n_views 3 --lambda_norm 0.01 --encoder_type VarTriPlaneEncoder --W 128 --iterations 30000 --max_num_pts 300000 -r 2 --load_time_step 0 --composition_rank 0
python render.py -s /media/STORAGE_4TB/NeRF_datasets/DTU/2DGS_data/dtu/DTU/scan110 -m ./output_rep/dtu/scan110/3views/SplatFields3D --pc_path ./output_rep/dtu/scan110/3views/3DGS/point_cloud/iteration_7000/point_cloud.ply --deform_weight 0 --white_background --lambda_mask 0.1 --n_views 3 --lambda_norm 0.01 --encoder_type VarTriPlaneEncoder --W 128 --iterations 30000 --max_num_pts 300000 -r 2 --load_time_step 0 --composition_rank 0

# ------------------------------------------------------------------------------

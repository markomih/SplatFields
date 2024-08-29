# Example script to run 3DGS, 3DGS with Moran loss, and SplatFields3D on the Blender dataset (Table 2 in the main paper and Tab C1 and C2 in the supplementary material)
set -x
SCENE=lego # any of the Blender scenes
N_VIEWS=10 # in range (4 6 8 10 12)
DATASET_ROOT=/media/STORAGE_4TB/NeRF_datasets/nerf_synthetic

# # to reproduce 3DGS
python train.py -s ${DATASET_ROOT}/${SCENE} --white_background --eval  -m ./output_rep/Blender/$SCENE/${N_VIEWS}views/3DGS --is_static --n_views $N_VIEWS --iterations 40000 --pts_samples hull --max_num_pts 300000 --load_time_step 0 --composition_rank 0
python render.py -s ${DATASET_ROOT}/${SCENE} --white_background --eval  -m ./output_rep/Blender/$SCENE/${N_VIEWS}views/3DGS --is_static --n_views $N_VIEWS --iterations 40000 --pts_samples hull --max_num_pts 300000 --load_time_step 0 --composition_rank 0

# to reproduce SplatFields3D
python train.py -s ${DATASET_ROOT}/${SCENE} --white_background --eval  -m ./output_rep/Blender/${SCENE}/${N_VIEWS}views/SplatFields --encoder_type VarTriPlaneEncoder --D 4 --lambda_norm 0.01 --test_iterations -1 --W 128 --n_views ${N_VIEWS} --iterations 40000 --pts_samples load --max_num_pts 100000 --pc_path ./output_rep/Blender/${SCENE}/${N_VIEWS}views/3DGS/point_cloud/iteration_40000/point_cloud.ply --load_time_step 0 --composition_rank 0
python render.py -s ${DATASET_ROOT}/${SCENE} --white_background --eval  -m ./output_rep/Blender/${SCENE}/${N_VIEWS}views/SplatFields --encoder_type VarTriPlaneEncoder --D 4 --lambda_norm 0.01 --test_iterations -1 --W 128 --n_views ${N_VIEWS} --iterations 40000 --pts_samples load --max_num_pts 100000 --pc_path ./output_rep/Blender/${SCENE}/${N_VIEWS}views/3DGS/point_cloud/iteration_40000/point_cloud.ply --load_time_step 0 --composition_rank 0

# to reproduce 3DGS w. L_moran
python train.py -s ${DATASET_ROOT}/${SCENE} --white_background --eval  -m ./output_rep/Blender/$SCENE/${N_VIEWS}views/3DGS_Lmoran --test_iterations -1 --is_static --n_views ${N_VIEWS} --iterations 40000 --pts_samples hull --max_num_pts 300000 --lambda_corr 0.01 --load_time_step 0 --composition_rank 0
python render.py -s ${DATASET_ROOT}/${SCENE} --white_background --eval  -m ./output_rep/Blender/$SCENE/${N_VIEWS}views/3DGS_Lmoran --test_iterations -1 --is_static --n_views ${N_VIEWS} --iterations 40000 --pts_samples hull --max_num_pts 300000 --lambda_corr 0.01 --load_time_step 0 --composition_rank 0


# to reproduce the ablation study (Table 3 in the main paper)
# 1st row: basic (MLP-only) model:
python train.py -s ${DATASET_ROOT}/${SCENE} --white_background --eval  -m ./output_rep/Blender/${SCENE}/${N_VIEWS}views/MLP --encoder_type none --test_iterations -1 --W 128 --n_views $N_VIEWS --iterations 40000 --pts_samples load --max_num_pts 300000 --pc_path ./output_rep/Blender/${SCENE}/${N_VIEWS}views/3DGS/point_cloud/iteration_40000/point_cloud.ply --load_time_step 0 --composition_rank 0
python render.py -s ${DATASET_ROOT}/${SCENE} --white_background --eval  -m ./output_rep/Blender/${SCENE}/${N_VIEWS}views/MLP --encoder_type none --test_iterations -1 --W 128 --n_views $N_VIEWS --iterations 40000 --pts_samples load --max_num_pts 300000 --pc_path ./output_rep/Blender/${SCENE}/${N_VIEWS}views/3DGS/point_cloud/iteration_40000/point_cloud.ply --load_time_step 0 --composition_rank 0
# 2nd row: basic (MLP-only) + L_2 norm:
python train.py -s ${DATASET_ROOT}/${SCENE} --white_background --eval  -m ./output_rep/Blender/${SCENE}/${N_VIEWS}views/MLP_norm0.01 --lambda_norm 0.01 --encoder_type none --test_iterations -1 --W 128 --n_views $N_VIEWS --iterations 40000 --pts_samples load --max_num_pts 300000 --pc_path ./output_rep/Blender/${SCENE}/${N_VIEWS}views/3DGS/point_cloud/iteration_40000/point_cloud.ply --load_time_step 0 --composition_rank 0
python render.py -s ${DATASET_ROOT}/${SCENE} --white_background --eval  -m ./output_rep/Blender/${SCENE}/${N_VIEWS}views/MLP_norm0.01 --lambda_norm 0.01 --encoder_type none --test_iterations -1 --W 128 --n_views $N_VIEWS --iterations 40000 --pts_samples load --max_num_pts 300000 --pc_path ./output_rep/Blender/${SCENE}/${N_VIEWS}views/3DGS/point_cloud/iteration_40000/point_cloud.ply --load_time_step 0 --composition_rank 0
# 3rd row: basic (MLP-only) + tri-CNN:
python train.py -s ${DATASET_ROOT}/${SCENE} --white_background --eval  -m ./output_rep/Blender/${SCENE}/${N_VIEWS}views/MLP_CNN --lambda_norm 0.01 --encoder_type VarTriPlaneEncoder --test_iterations -1 --W 128 --n_views $N_VIEWS --iterations 40000 --pts_samples load --max_num_pts 300000 --pc_path ./output_rep/Blender/${SCENE}/${N_VIEWS}views/3DGS/point_cloud/iteration_40000/point_cloud.ply --load_time_step 0 --composition_rank 0
python render.py -s ${DATASET_ROOT}/${SCENE} --white_background --eval  -m ./output_rep/Blender/${SCENE}/${N_VIEWS}views/MLP_CNN --lambda_norm 0.01 --encoder_type VarTriPlaneEncoder --test_iterations -1 --W 128 --n_views $N_VIEWS --iterations 40000 --pts_samples load --max_num_pts 300000 --pc_path ./output_rep/Blender/${SCENE}/${N_VIEWS}views/3DGS/point_cloud/iteration_40000/point_cloud.ply --load_time_step 0 --composition_rank 0
# 4th row: full model (MLP+L_2 norm+tri-CNN):
python train.py -s ${DATASET_ROOT}/${SCENE} --white_background --eval  -m ./output_rep/Blender/${SCENE}/${N_VIEWS}views/MLP_norm0.01_CNN --lambda_norm 0.01 --encoder_type VarTriPlaneEncoder --test_iterations -1 --W 128 --n_views $N_VIEWS --iterations 40000 --pts_samples load --max_num_pts 300000 --pc_path ./output_rep/Blender/${SCENE}/${N_VIEWS}views/3DGS/point_cloud/iteration_40000/point_cloud.ply --load_time_step 0 --composition_rank 0
python render.py -s ${DATASET_ROOT}/${SCENE} --white_background --eval  -m ./output_rep/Blender/${SCENE}/${N_VIEWS}views/MLP_norm0.01_CNN --lambda_norm 0.01 --encoder_type VarTriPlaneEncoder --test_iterations -1 --W 128 --n_views $N_VIEWS --iterations 40000 --pts_samples load --max_num_pts 300000 --pc_path ./output_rep/Blender/${SCENE}/${N_VIEWS}views/3DGS/point_cloud/iteration_40000/point_cloud.ply --load_time_step 0 --composition_rank 0

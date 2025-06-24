dataset=emidec_diff_seg
types=1
diffusion_img_size=72
diffusion_depth_size=1
diffusion_num_channels=1
train_num_steps=55001
cond_dim=16
batch_size=4
seg_num_classes=4
seg_lr=0.01
seg_weight_decay=0.00003
seg_num_epochs=5000
seg_optimizer_name=SGD
seg_network_type=nnUNet
seg_image_width=64
seg_image_height=64
jump_length=2
jump_n_sample=2
gpus=0
seg_nnUNet_plan_file="/home/farheen/E/Scar_Syhthesis/LeFusion/LeFusion-main/nnUNet/nnUNet_preprocessed/Dataset111_EMIDEC/nnUNetPlans.json"
seg_pretrained_weights="/home/farheen/E/Scar_Syhthesis/LeFusion/LeFusion-main/nnUNet/nnUNet_results/Dataset111_EMIDEC/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
dataset_root_dir_path=data/EMIDEC/train_emidec_dataset_slices/Pathological
dataset_root_dir_norm=data/EMIDEC/train_emidec_dataset_slices/Normal_t
dataset_root_dir_valid=data/EMIDEC/train_emidec_dataset_slices/Normal_v
results_folder=LeFusion/LeFusion_Model_Joint/EMIDEC
seg_results_folder=LeFusion/LeFusion_Model_Joint/EMIDEC/nnUNet_results


export nnUNet_raw="/home/farheen/E/Scar_Syhthesis/LeFusion/LeFusion-main/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/home/farheen/E/Scar_Syhthesis/LeFusion/LeFusion-main/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/farheen/E/Scar_Syhthesis/LeFusion/LeFusion-main/nnUNet/nnUNet_results"
echo ${nnUNet_raw}
echo ${nnUNet_preprocessed}
echo ${nnUNet_results}


python LeFusion/train/train_diff_seg.py \
    dataset=$dataset \
    model.types=$types \
    model.diffusion_img_size=$diffusion_img_size \
    model.diffusion_depth_size=$diffusion_depth_size \
    model.diffusion_num_channels=$diffusion_num_channels \
    model.batch_size=$batch_size \
    model.cond_dim=$cond_dim \
    model.train_num_steps=$train_num_steps \
    model.seg_num_epochs=$seg_num_epochs \
    model.batch_size=$batch_size \
    model.seg_lr=$seg_lr \
    model.seg_image_width=$seg_image_width \
    model.seg_image_height=$seg_image_height \
    model.gpus=$gpus \
    model.seg_weight_decay=$seg_weight_decay \
    model.seg_num_classes=$seg_num_classes \
    model.seg_optimizer_name=$seg_optimizer_name \
    model.seg_network_type=$seg_network_type \
    dataset.root_dir_path=$dataset_root_dir_path \
    dataset.root_dir_norm=$dataset_root_dir_norm \
    dataset.root_dir_valid=$dataset_root_dir_valid \
    model.results_folder=$results_folder \
    model.seg_results_folder=$seg_results_folder \
    schedule_jump_params.jump_length=$jump_length \
    schedule_jump_params.jump_n_sample=$jump_n_sample \

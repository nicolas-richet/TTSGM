GPU_ID=0
DATASET=tiny-imagenet
DATA_DIR=./data/Tiny-Imagenet
BATCH_SIZE=128
LR=1e-4
METHOD=ttsgm

# TTSGM parameters
ITERATIONS=10
SPARSITY=0.999
MULTI_TEMPLATE_SIGN_METHOD=mean_sign
MULTI_TEMPLATE_MASK_METHOD=mean_abs
NB_TEMPLATES=8

WATT_TEMPLATE_DIR=./templates.yaml
TRIALS=3
ALL_CORRUPTIONS="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
ENCODER=vision


# single-template
BACKBONE=ViT-B/32
SAVE_DIR=./save/${DATASET}/${DOMAIN}/${BATCH_SIZE}/${BACKBONE}/ttsgm-s${SPARSITY}-${ENCODER}-lr${LR}-mask${MULTI_TEMPLATE_MASK_METHOD}-${MULTI_TEMPLATE_SIGN_METHOD}-TPS${NB_TEMPLATES}
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --trials $TRIALS --data_dir $DATA_DIR --dataset $DATASET --adapt --method $METHOD --corruptions_list $ALL_CORRUPTIONS --save_dir $SAVE_DIR --backbone $BACKBONE --batch-size $BATCH_SIZE --lr $LR --watt_temps $WATT_TEMPLATE_DIR --iterations $ITERATIONS --sparsity $SPARSITY --encoder $ENCODER --target_type $TARGET_TYPE --multi_template_sign_method $MULTI_TEMPLATE_SIGN_METHOD --multi_template_mask_method $MULTI_TEMPLATE_MASK_METHOD --nb_templates $NB_TEMPLATES

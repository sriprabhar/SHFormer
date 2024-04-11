MODEL='<model name or experiment name>'
BASE_PATH='<base path>'
BATCH_SIZE=1
DEVICE='cuda:0'
USMASK_PATH=${BASE_PATH}'/usmasks/'

for DATASET_TYPE in 'mrbrain_t1' #'mrbrain_flair' 'ixi_pd' 'ixi_t2'
    do
    for MASK_TYPE in 'cartesian' #'gaussian'
        do 
        for ACC_FACTOR in '4x' #'5x' '8x'
            do 
            echo ${DATASET_TYPE}','${MASK_TYPE}','${ACC_FACTOR} 
            CHECKPOINT=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/best_model.pt'
            OUT_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/results'
            DATA_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/validation/acc_'${ACC_FACTOR}
            echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --mask_type ${MASK_TYPE} 
            python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --mask_type ${MASK_TYPE} --usmask_path ${USMASK_PATH}
            done 
        done 
    done




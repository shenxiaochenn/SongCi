export MODEL_FLAGS_128="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
export TRAIN_FLAGS="--lr 1e-4 --batch_size 2"
export MODEL_FLAGS_256="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
export NUM_GPUS=8
MODEL_FLAGS="--attention_resolutions 32,16,8 --diffusion_steps 1000 --large_size 256  --small_size 128 --learn_sigma True --noise_schedule linear --num_channels 192 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --feat_cond False"

#mpiexec -n $NUM_GPUS python scripts/super_res_train.py --data_dir gen_data.txt --log_interval 100 $MODEL_FLAGS $TRAIN_FLAGS
mpiexec -n $NUM_GPUS python scripts/image_train.py --data_dir gen_data.txt $MODEL_FLAGS_128 $TRAIN_FLAGS

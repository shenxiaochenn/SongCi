export MODEL_FLAGS_128="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
export TRAIN_FLAGS="--lr 1e-4 --batch_size 4"
export NUM_GPUS=1
#MODEL_FLAGS="--attention_resolutions 32,16,8 --diffusion_steps 4000 --large_size 256  --small_size 128 --learn_sigma True --noise_schedule linear --num_channels 192 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
#MODEL_FLAGS="--attention_resolutions 32,16,8 --diffusion_steps 1000 --large_size 256  --small_size 128 --learn_sigma True --noise_schedule linear --num_channels 192 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --feat_cond False"

#CUDA_VISIBLE_DEVICES=7 mpiexec -n $NUM_GPUS  python scripts/image_sample.py $MODEL_FLAGS_128  --timestep_respacing 250
#CUDA_VISIBLE_DEVICES=1 mpiexec -n $NUM_GPUS  python scripts/super_res_sample.py $MODEL_FLAGS --timestep_respacing 250
CUDA_VISIBLE_DEVICES=1 mpiexec -n $NUM_GPUS  python scripts/image_sample_prototype.py $MODEL_FLAGS_128  --timestep_respacing 250

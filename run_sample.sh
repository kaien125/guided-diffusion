#!/bin/bash
#SBATCH --job-name=diff__ori_sample
#SBATCH --output=output_guided_diffusion_origin_50000_classify_samples.txt
#SBATCH --error=error_guided_diffusion_origin_50000_classify_samples.err
#SBATCH --partition=SCT

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G

#SBATCH --qos=normal

#SBATCH --mail-type=ALL
#SBATCH --mail-user=s3815738@rmit.edu.au

cd ~/code/guided-diffusion/scripts

export PYTHONPATH=export PYTHONPATH=/opt/home/s3815738/code/guided-diffusion:/usr/lib/python38.zip:/usr/lib/python3.8:/usr/lib/python3.8/lib-dynload:/opt/home/s3815738/env/venv/lib/python3.8/site-packages

SAMPLE_FLAGS="--batch_size 64 --num_samples 50000 --timestep_respacing 250"

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"

python classifier_sample.py $MODEL_FLAGS --classifier_scale 1.0 --classifier_path ../models/64x64_classifier.pt --classifier_depth 4 --model_path ../models/64x64_diffusion.pt $SAMPLE_FLAGS > classifier_sample_output.txt

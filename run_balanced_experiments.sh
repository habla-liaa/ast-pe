#!/bin/bash
set -x

export LC_NUMERIC="en_US.UTF-8"
export PYTHONHASHSEED=1234

WARMUP='global/warmup_steps=10000'
LOGS1='global/checkpoint_time_unit=epoch'
LOGS2='global/checkpoint_freq=1'
LOGS3='global/validation_data=AudiosetBalancedValidationGenerator->out'
LOGS4='global/validation_freq=1'
LOGS5='global/validation_time_unit=epoch'
#Abs PE
paiprun configs/main/balanced_audioset_pretraining.yaml --output_path experiments/vit/aset_balanced_abs_learnt_pe-2 --mods "global/experiment_name=audioset_balanced_abs_learnt_pe&global/max_lr=0.00025&$WARMUP&global/batch_size=62"
#PEG
#paiprun configs/main/balanced_audioset_pretraining.yaml --output_path experiments/vit/aset_balanced_peg --mods "global/experiment_name=audioset_balanced_peg&global/max_lr=0.00025&global/dienen_config=!yaml configs/model/architectures/audio_spectrogram_transformer_peg.yaml&$WARMUP&$LOGS1&$LOGS2&$LOGS3&$LOGS4&$LOGS5&global/batch_size=62"
#Alibi2D Time + Abs Freq
#paiprun configs/main/balanced_audioset_pretraining.yaml --output_path experiments/vit/aset_balanced_alibitime_freqabs --mods "global/experiment_name=audioset_balanced_alibitime&global/max_lr=0.00025&global/dienen_config=!yaml configs/model/architectures/audio_spectrogram_transformer_alibitime_absfreq.yaml&$WARMUP&$LOGS1&$LOGS2&$LOGS3&$LOGS4&$LOGS5&global/batch_size=62"
#Alibi2D
#paiprun configs/main/balanced_audioset_pretraining.yaml --output_path experiments/vit/aset_balanced_alibi2d --mods "global/experiment_name=audioset_balanced_alibi2d&global/max_lr=0.00025&global/dienen_config=!yaml configs/model/architectures/audio_spectrogram_transformer_alibi2d_zeroinit.yaml&$WARMUP&$LOGS1&$LOGS2&$LOGS3&$LOGS4&$LOGS5&global/batch_size=62"
#NO PE
#paiprun configs/main/balanced_audioset_pretraining.yaml --output_path experiments/vit/aset_balanced_nope --mods "global/experiment_name=audioset_balanced_nope&global/max_lr=0.00025&global/dienen_config=!yaml configs/model/architectures/audio_spectrogram_transformer_no_pe.yaml&$WARMUP&$LOGS1&$LOGS2&$LOGS3&$LOGS4&$LOGS5&global/batch_size=62"
#HUANG
#paiprun configs/main/balanced_audioset_pretraining.yaml --output_path experiments/vit/aset_balanced_huang --mods "global/experiment_name=audioset_balanced_huang&global/max_lr=0.00025&global/dienen_config=!yaml configs/model/architectures/audio_spectrogram_transformer_huang2d.yaml&$WARMUP&$LOGS1&$LOGS2&$LOGS3&$LOGS4&$LOGS5&global/batch_size=24"
#PEG+ABS
#paiprun configs/main/balanced_audioset_pretraining.yaml --output_path experiments/vit/aset_balanced_pegabs --mods "global/experiment_name=audioset_balanced_pegabs&global/max_lr=0.00025&global/dienen_config=!yaml configs/model/architectures/audio_spectrogram_transformer_peg_plus_abs.yaml&$WARMUP&$LOGS1&$LOGS2&$LOGS3&$LOGS4&$LOGS5&global/batch_size=62"
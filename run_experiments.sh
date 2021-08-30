#!/bin/bash
set -x

export LC_NUMERIC="en_US.UTF-8"
export PYTHONHASHSEED=1234

#Absolute learned positional embeddings:
#paiprun configs/main/audioset_pretraining.yaml --output_path experiments/vit/abs_learnt_pe --mods "global/experiment_name=audioset_abs_learnt_pe&global/temp_dataset_paths=!nocache ['~/temp2/1','~/temp2/2']"
#PEG
paiprun configs/main/audioset_pretraining.yaml --output_path experiments/vit/peg_5_layers --mods "global/experiment_name=audioset_peg&global/dienen_config=!yaml configs/model/architectures/audio_spectrogram_transformer_peg.yaml"
#!/bin/bash
set -x

export LC_NUMERIC="en_US.UTF-8"
export PYTHONHASHSEED=1234

#Absolute learned positional embeddings:
#paiprun configs/main/audioset_pretraining.yaml --output_path experiments/vit/abs_learnt_pe --mods "global/experiment_name=audioset_abs_learnt_pe&global/temp_dataset_paths=!nocache ['~/temp2/1','~/temp2/2']&global/max_lr=0.0005&Tasks/AudiosetModel/task_hash=29ee2dbd2ca67d07ea6bdc598b0a17352309a770"
#PEG
#paiprun configs/main/audioset_pretraining.yaml --output_path experiments/vit/peg_5_layers --mods "global/experiment_name=audioset_peg&global/dienen_config=!yaml configs/model/architectures/audio_spectrogram_transformer_peg.yaml&global/max_lr=0.0005&Tasks/AudiosetModel/task_hash=d62760c2ef31d621ca8ae8f656e3ca61e590c42a"
#Huang2D
paiprun configs/main/audioset_pretraining.yaml --output_path experiments/vit/huang2d --mods "global/experiment_name=audioset_huang2d&global/dienen_config=!yaml configs/model/architectures/audio_spectrogram_transformer_huang2d.yaml&global/max_lr=0.0005&global/temp_dataset_paths=!nocache ['~/temp2/1','~/temp2/2']&global/batch_size=32&global/steps_per_epoch=51750"
#Alibi2D
#paiprun configs/main/audioset_pretraining.yaml --output_path experiments/vit/alibi2d --mods "global/experiment_name=audioset_alibi2d&global/dienen_config=!yaml configs/model/architectures/audio_spectrogram_transformer_alibi2d.yaml&global/max_lr=0.0005"
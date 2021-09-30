#!/bin/bash
set -x

export LC_NUMERIC="en_US.UTF-8"
export PYTHONHASHSEED=1234

#Absolute learned positional embeddings:
#paiprun configs/main/audioset_pretraining.yaml --output_path experiments/vit/abs_learnt_pe --mods "global/experiment_name=audioset_abs_learnt_pe&global/temp_dataset_paths=!nocache ['~/temp2/1','~/temp2/2']&global/max_lr=0.0005&Tasks/AudiosetModel/task_hash=29ee2dbd2ca67d07ea6bdc598b0a17352309a770"
#PEG
#paiprun configs/main/audioset_pretraining.yaml --output_path experiments/vit/peg_5_layers --mods "global/experiment_name=audioset_peg&global/dienen_config=!yaml configs/model/architectures/audio_spectrogram_transformer_peg.yaml&global/max_lr=0.0005&Tasks/AudiosetModel/task_hash=d62760c2ef31d621ca8ae8f656e3ca61e590c42a"
#Huang2D
#paiprun configs/main/audioset_pretraining.yaml --output_path experiments/vit/huang2d --mods "global/experiment_name=audioset_huang2d&global/dienen_config=!yaml configs/model/architectures/audio_spectrogram_transformer_huang2d.yaml&global/max_lr=0.0005&global/batch_size=32&global/steps_per_epoch=51750&Tasks/AudiosetModel/task_hash=779a6fb2d26e064c5c3809be176886d5dd628355"
#Alibi2D
#paiprun configs/main/audioset_pretraining.yaml --output_path experiments/vit/alibi2d --mods "global/experiment_name=audioset_alibi2d&global/dienen_config=!yaml configs/model/architectures/audio_spectrogram_transformer_alibi2d.yaml&global/max_lr=0.0005"

#PEG constant LR:
#paiprun configs/main/audioset_pretraining.yaml --output_path experiments/vit/peg_5_layers_constantlr --mods "global/experiment_name=audioset_peg_constantlr&global/dienen_config=!yaml configs/model/architectures/audio_spectrogram_transformer_peg_constantlr.yaml"

#PEG with balanced sampling
#paiprun configs/main/audioset_pretraining_balancedsampling.yaml --output_path experiments/vit/peg_5_layers_mixup --mods "global/experiment_name=audioset_peg_mixup&global/dienen_config=!yaml configs/model/architectures/audio_spectrogram_transformer_peg2.yaml&global/max_lr=0.0005&global/do_balanced_sampling=True&Tasks/AudiosetModel/task_hash=7dae0a038315ba961f1659a2c54c04982b8f088e"

#PEG2 with balanced sampling
#paiprun configs/main/audioset_pretraining_balancedsampling.yaml --output_path experiments/vit/peg_5_layers_mixup_2 --mods "global/experiment_name=audioset_peg_mixup2&global/dienen_config=!yaml configs/model/architectures/audio_spectrogram_transformer_peg3.yaml&global/do_balanced_sampling=True&global/temp_dataset_paths=!nocache ['~/temp2/1','~/temp2/2']"

#PEG original with balanced sampling
#paiprun configs/main/audioset_pretraining_balancedsampling.yaml --output_path experiments/vit/peg_5_layers_balanced --mods "global/experiment_name=audioset_peg_balanced&global/dienen_config=!yaml configs/model/architectures/audio_spectrogram_transformer_peg.yaml&global/max_lr=0.0005&global/do_balanced_sampling=True"

#PEG original with balanced sampling using lr decay
#paiprun configs/main/audioset_pretraining_balancedsampling.yaml --output_path experiments/vit/peg_5_layers_balanced_lrdecay --mods "global/experiment_name=audioset_peg_balanced_lrdecay&global/dienen_config=!yaml configs/model/architectures/audio_spectrogram_transformer_peg_lrdecay.yaml&global/max_lr=0.0005&global/do_balanced_sampling=True"

#PEG + absolute pe
#paiprun configs/main/audioset_pretraining.yaml --output_path experiments/vit/peg_5_layers_plus_abs --mods "global/experiment_name=audioset_peg_plus_abs&global/dienen_config=!yaml configs/model/architectures/audio_spectrogram_transformer_peg_plus_abs.yaml&global/max_lr=0.0005"

#No PE
#paiprun configs/main/audioset_pretraining.yaml --output_path experiments/vit/no_pe --mods "global/experiment_name=audioset_no_pe&global/dienen_config=!yaml configs/model/architectures/audio_spectrogram_transformer_no_pe.yaml&global/max_lr=0.0005"

#Alibi2D (round 2: cls token correction)
#paiprun configs/main/audioset_pretraining.yaml --output_path experiments/vit/alibi2d_zeroinit --mods "global/experiment_name=audioset_alibi2d_zeroinit&global/dienen_config=!yaml configs/model/architectures/audio_spectrogram_transformer_alibi2d_zeroinit.yaml&global/max_lr=0.0005&global/temp_dataset_paths=!nocache ['~/temp2/1','~/temp2/2']"

#Alibi2D (Time only heads) + Abs2D (Freq only)
paiprun configs/main/audioset_pretraining.yaml --output_path experiments/vit/alibi_time_abs_freq --mods "global/experiment_name=audioset_alibi_time_abs_freq&global/dienen_config=!yaml configs/model/architectures/audio_spectrogram_transformer_alibitime_absfreq.yaml&global/max_lr=0.0005&global/temp_dataset_paths=!nocache ['~/temp2/1','~/temp2/2']"
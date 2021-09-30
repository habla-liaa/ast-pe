#!/bin/bash
set -x

export LC_NUMERIC="en_US.UTF-8"
export PYTHONHASHSEED=1234

#PEG-SWA pretraining
#paiprun configs/main/esc50_training.yaml --output_path experiments/esc50/peg_swa --mods "global/experiment_name=esc_peg_swa&global/pretrained_ast=pretrained_models/peg-embeddings-swa.dnn"

#ABS-SWA pretraining
#paiprun configs/main/esc50_training.yaml --output_path experiments/esc50/abs_swa --mods "global/experiment_name=esc_abs_swa&global/pretrained_ast=pretrained_models/abs-pe-encoding-swa.dnn"

#PEG-PLUS-ABS pretraining
#paiprun configs/main/esc50_training.yaml --output_path experiments/esc50/peg_plus_abs_swa --mods "global/experiment_name=esc_peg_plus_abs_swa&global/pretrained_ast=pretrained_models/peg-plus-abs-swa.dnn"

#NO-PE pretraining
paiprun configs/main/esc50_training.yaml --output_path experiments/esc50/no_pe_swa --mods "global/experiment_name=esc_no_pe_swa&global/pretrained_ast=pretrained_models/ast-no-pe-swa.dnn"
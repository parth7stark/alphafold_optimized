#!/bin/bash

JOB_NAME="${1}"
MACHINE_NAME="${2}"
FASTA_FILES="${3}"
MAX_TEMPLATE_DATE="${4:-2023-01-01}"
USE_PRECOMPUTED_MSAS="${5:-true}"
MODEL_PRESET="${6:-monomer}"
DB_PRESET="${7:-reduced_dbs}"
NUM_PREDS_PER_MODEL="${8:-5}"
MODELS_TO_RELAX="${9:-all}"
GPU_DEVICES="${10:-all}"
PERFORM_MD_ONLY="${11:-false}"
USE_AMBER="${12:-true}"
USE_DROPOUT="${13:-false}"
USE_BFLOAT16="${14:-true}"
USE_FUSE="${15:-false}"
MAX_SEQ="${16:-512}"
MAX_EXTRA_SEQ="${17:-5120}"
NUM_ENSEMBLE="${18:-1}"
NUM_RECYCLES="${19:-3}"
RECYCLE_EARLY="${20:-0.5}"

echo "$JOB_NAME is submitted for Alphafold structure prediction..."

if [ -z "$1" ]; then
  echo "Useage: submit_dgx.sh JOB_NAME MACHINE_NAME FASTA_FILES MAX_TEMPLATE_DATE USE_PRECOMPUTED_MSAS MODEL_PRESET DB_PRESET NUM_PREDS_PER_MODEL MODELS_TO_RELAX GPU_DEVICES"
  echo "MACHINE_NAME can be dgx or dgx_short"
  echo "FASTA_FILES should be /path/to/directory/file_name.fasta"
  echo "MAX_TEMPLATE_DATE should be YYYY-MM-DD format"
  echo "USE_PRECOMPUTED_MSAS can be true or false"
  echo "MODEL_PRESET can be monomer or multimer"
  echo "DB_PRESEET can be full_dbs or reduced_dbs"
  echo "NUM_PREDS_PER_MODEL should be an integer... affects when MODEL_PRESET=multimer"
  echo "MODELS_TO_RELAX can be all, best, or none"
  echo "GPU_DEVICES can be all or comma seperated integers"
  echo "PERFORM_MD_ONLY is true when whole prediction is done in one-go; if MD is done separately, set to false"
  echo "USE_AMBER is true when using Amber force fields, else Charmm"
  echo "USE_DROPOUT is false by default, affecting neural network diversity"
  echo "USE_BFLOAT16 is true by default, affecting inference speed"
  echo "USE_FUSE is false by default, affecting (potentially) inference speed"
  echo "MAX_SEQ is None by default, hence using original MSA/template scheme"
  echo "MAX_EXTRA_SEQ is None by default, hence using original MSA/template scheme"
  echo "NUM_ENSEMBLE is 1 by default, CASP 14 standard"
  echo "NUM_RECYCLES is None by default, hence recycling 3 times"
  echo "RECYCLE_EARLY is 0.5 by default"
  exit
fi

echo "JOB_NAME $1; MACHINE_NAME $2; FASTA_FILES $3; MAX_TEMPLATE_DATE $4; USE_PRECOMPUTED_MSAS $5; MODEL_PRESET $6; DB_PRESEET $7; NUM_PREDS_PER_MODEL $8; MODELS_TO_RELAX $9; GPU_DEVICES ${10}; PERFORM_MD_ONLY ${11}; USE_AMBER ${12} USE_DROPOUT ${13} USE_BFLOAT16 ${14} USE_FUSE ${15} MAX_SEQ ${16} MAX_EXTRA_SEQ ${17} NUM_ENSEMBLE ${18} NUM_RECYCLES ${19} are chosen..."

PYTHON=/Projects/ghaemi/Programs/Conda/envs/AF/bin/python3
ALPHAFOLD_DIR=/Scr/hyunpark/Alphafold
TCBG_DIR=/Scr/alpha_fold
DOCKER_NAME=hyunp2/alphafold_original

qsub -q $MACHINE_NAME -N $JOB_NAME -j y -o ~/Jobs << EOF
pushd $TCBG_DIR/alphafold

$PYTHON $TCBG_DIR/alphafold/docker/run_docker.py --data_dir=$TCBG_DIR/data_hyun_official --fasta_paths=$FASTA_FILES --max_template_date=$MAX_TEMPLATE_DATE --use_precomputed_msas=$USE_PRECOMPUTED_MSAS --model_preset=$MODEL_PRESET --db_preset=$DB_PRESET --num_multimer_predictions_per_model=$NUM_PREDS_PER_MODEL --enable_gpu_relax=true --output_dir=$TCBG_DIR/outputs --docker_image_name=$DOCKER_NAME --models_to_relax=$MODELS_TO_RELAX --gpu_devices=$GPU_DEVICES --perform_MD_only=$PERFORM_MD_ONLY --use_amber=$USE_AMBER --use_dropout=$USE_DROPOUT --use_bfloat16=$USE_BFLOAT16 --use_fuse=$USE_FUSE --max_seq=$MAX_SEQ --max_extra_seq=$MAX_EXTRA_SEQ --num_ensemble=$NUM_ENSEMBLE --num_recycles=$NUM_RECYCLES --recycle_early_stop_tolerance=$RECYCLE_EARLY

EOF



#JOBNAME="${1/%.namd/}"

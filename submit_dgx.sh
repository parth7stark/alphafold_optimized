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
CONTINUED_SIMULATION="${11:-true}"

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
  echo "CONTINUED_SIMULATION is true when whole prediction is done in one-go; if MD is done separately, set to false"
  exit
fi

echo "JOB_NAME $1; MACHINE_NAME $2; FASTA_FILES $3; MAX_TEMPLATE_DATE $4; USE_PRECOMPUTED_MSAS $5; MODEL_PRESET $6; DB_PRESEET $7; NUM_PREDS_PER_MODEL $8; MODELS_TO_RELAX $9; GPU_DEVICES ${10}; CONTINUED_SIMULATION ${11} are chosen..."

PYTHON=/Projects/ghaemi/Programs/Conda/envs/AF/bin/python3
ALPHAFOLD_DIR=/Scr/hyunpark/Alphafold
TCBG_DIR=/Scr/alpha_fold
DOCKER_NAME=hyunp2/alphafold_original

qsub -q $MACHINE_NAME -N $JOB_NAME -j y -o ~/Jobs << EOF
pushd $ALPHAFOLD_DIR/alphafold

$PYTHON $ALPHAFOLD_DIR/alphafold/docker/run_docker.py --data_dir=$TCBG_DIR/data_hyun_official --fasta_paths=$FASTA_FILES --max_template_date=$MAX_TEMPLATE_DATE --use_precomputed_msas=$USE_PRECOMPUTED_MSAS --model_preset $MODEL_PRESET --db_preset $DB_PRESET --num_multimer_predictions_per_model $NUM_PREDS_PER_MODEL --enable_gpu_relax=true --output_dir=$TCBG_DIR/outputs --docker_image_name $DOCKER_NAME --models_to_relax=$MODELS_TO_RELAX --gpu_devices=$GPU_DEVICES --continued_simulation=$CONTINUED_SIMULATION
EOF

#JOBNAME="${1/%.namd/}"

#!/bin/bash

SWEEP_ID=$1
N_RUNS=300

date;hostname;pwd;

vipu create partition p${SLURM_JOB_ID} --allocation c${SLURM_JOB_ID} --size 16 --reconfigurable

#extract the API HOST from the config file
export IPUOF_VIPU_API_HOST=$(grep api-host /etc/vipu/vipu-cli.hcl | sed -e 's/api-host=\(.*\)$/\1/g' -e 's/\"//g' )

export IPUOF_VIPU_API_PARTITION_ID=p${SLURM_JOB_ID}

source /nethome/samuelm/git/SDKs/poplar_sdk-ubuntu_20_04-3.3.0+1403-208993bbb7/enable # <-------- CHANGE THIS
source /nethome/samuelm/git/ENVs/3.3.0+1403/3.3.0+1403_poptorch/bin/activate # <-------- CHANGE THIS

echo $SWEEP_ID
if [ -z "${SWEEP_ID}" ];
then
  echo "Environment variable SWEEP_ID must be set.  Exiting..."
  exit
else
  echo "Running wandb sweep with id: ${SWEEP_ID}"
fi

LOGGING_DIR="/nethome/samuelm/git/paper_experiments/graphium/runs/${SLURM_JOBID}" # <-------- CHANGE THIS (run from here!)

export WANDB_NOTES="$(hostname)"

export POPLAR_LOG_LEVEL=INFO
# ensure cache is set
export TF_POPLAR_FLAGS=' --executable_cache_path=/nethome/samuelm/poplar_cache/' # <-------- CHANGE THIS

# if [ ! -d "/localdata/neurips2023-large/" ]
# then
#     echo "Directory /localdata/neurips2023-large/ DOES NOT exists. Creating..."
#     mkdir /localdata/neurips2023-large/
# fi

# if [ ! -d "/localdata/neurips2023-large/a369011e12e531e24c916bf2a0fe9e39" ]
# then
#     echo "Directory /localdata/neurips2023-large/a369011e12e531e24c916bf2a0fe9e39 DOES NOT exists. Copying..."
#     cp -r /nethome/kerstink/datacache/neurips2023-large/*e12e531e24c916bf2a0fe9e39 /localdata/neurips2023-large/
# fi
# -----------------
if [ ! -d "/localdata/neurips2023-large/" ]
then
    echo "Directory /localdata/neurips2023-large/ DOES NOT exists. Creating..."
    mkdir /localdata/neurips2023-large/
fi

if [ ! -d "/localdata/neurips2023-large/g25/" ]
then
    echo "Directory /localdata/neurips2023-large/g25/ DOES NOT exists. Copying..."
    mkdir /localdata/neurips2023-large/g25/
fi

if [ ! -d "/localdata/neurips2023-large/g25/b8a504ed7ce403aace5a7490e671f77c" ]
then
    echo "Directory /localdata/neurips2023-large/g25/b8a504ed7ce403aace5a7490e671f77c DOES NOT exists. Copying..."
    cp -r /nethome/kerstink/datacache/neurips2023-large/g25/*504ed7ce403aace5a7490e671f77c /localdata/neurips2023-large/g25/
fi
# -----------------
echo "Entering repo"
REPO="/nethome/samuelm/git/paper_experiments/graphium/" # <-------- CHANGE THIS
cd $REPO || exit

# pip3 install -r requirements.txt

FILENAME=$(echo "${SWEEP_ID}" | tr '/' '_')

echo $LOGGING_DIR
echo $SWEEP_ID
echo $FILENAME
echo $N_RUNS

mkdir -p $LOGGING_DIR

wandb agent --count=${N_RUNS} ${SWEEP_ID} > $LOGGING_DIR/${FILENAME}_agent1.txt 2>&1

wait

#!/bin/bash -l
#SBATCH --time=6:00:00                        ## wallclock time hh:mm:ss, change according to the need
#SBATCH --gres=gpu                            ## --gres=gpu:n, where n denotes how many gpu you require 
#SBATCH --mem-per-cpu=30000                   ## 10G of memory, change according to the need
#SBATCH --array=10                             ## creates an array of 5´4 jobs (tasks) with index values 0, 1, 2, 3.
#SBATCH --output=WR_femnist_%a.out
#SBATCH --mail-user=yujia.guo@aalto.fi
#SBATCH --mail-type=END

module purge                                                      # unload the anaconda module, or previously loaded models
source /scratch/cs/ssg-mlsec/miniconda3/etc/profile.d/conda.sh    # load our miniconda
conda activate pysyft                                             # load pysyft environment

python main.py --config_file configurations/WR/femnist/100_20${SLURM_ARRAY_TASK_ID}.ini  --experiment training


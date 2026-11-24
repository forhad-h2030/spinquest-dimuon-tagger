#!/bin/bash
module purge
module load gcc/14.2.0
module load openmpi/5.0.7
module load python/3.13.11
module load apptainer/1.5.0
module load root/6.38.00
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"

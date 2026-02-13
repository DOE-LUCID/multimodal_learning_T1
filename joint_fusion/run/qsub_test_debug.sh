#!/bin/bash -l

#PBS -A GeomicVar
#PBS -l walltime=01:00:00
#file systems used by the job
#PBS -l filesystems=home:eagle


#PBS -N multimodal_learning

#PBS -q debug

# Controlling the output of your application
# UG Sec 3.3 page UG-42 Managing Output and Error Files
# By default, PBS spools your output on the compute node and then uses scp to move it the
# destination directory after the job finishes.  Since we have globally mounted file systems
# it is highly recommended that you use the -k option to write directly to the destination
# the doe stands for direct, output, error
#PBS -k doe
#PBS -o /grand/GeomicVar/jozhw/multimodal_learning_T1/tmp
#PBS -e /grand/GeomicVar/jozhw/multimodal_learning_T1/tmp

# If you want to merge stdout and stderr, use the -j option
# oe=merge stdout/stderr to stdout, eo=merge stderr/stdout to stderr, n=don't merge
#PBS -j n

# Controlling email notifications
# UG Sec 2.5.1, page UG-25 Specifying Email Notification
# When to send email b=job begin, e=job end, a=job abort, j=subjobs (job arrays), n=no mail
#PBS -m be
# Be default, mail goes to the submitter, use this option to add others (uncomment to use)
#PBS -M johnzhouyangwu@hsph.harvard.edu

# Setting job dependencies
# UG Section 6.2, page UG-109 Using Job Dependencies
# There are many options for how to set up dependencies;  afterok will give behavior similar
# to Cobalt (uncomment to use)
##PBS depend=afterok:<jobid>:<jobid>

# Environment variables (uncomment to use)
# UG Section 6.12, page UG-126 Using Environment Variables
# RG Sect 2.57.7, page RG-233 Environment variables PBS puts in the job environment
##PBS -v <variable list>
## -v a=10, "var2='A,B'", c=20, HOME=/home/zzz
##PBS -V exports all the environment variables in your environment to the compute node

echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

# load conda environment and change working directory
module use /soft/modulefiles
module load conda
conda activate /grand/GeomicVar/embeddings_for_john/multimodal_env

echo Jobid: $PBS_JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`

./joint_fusion/run/run_testing.sh

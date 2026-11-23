#!/bin/bash

# CONFIG="joint_fusion/config/base_config.yaml"
CONFIG="/lus/eagle/clone/g2/projects/GeomicVar/jozhw/multimodal_learning_T1/checkpoints/checkpoint_2026-02-14-05-44-19/config.yaml"

# make sure to call from root dir
python -m joint_fusion.testing.interpret_omics --config=$CONFIG --top-n=20

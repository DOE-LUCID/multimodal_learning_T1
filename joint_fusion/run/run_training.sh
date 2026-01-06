#!/bin/bash

CONFIG="joint_fusion/config/base_config.yaml"

# make sure to call from root dir
python -m joint_fusion.training.trainer --config=$CONFIG

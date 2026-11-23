#!/bin/bash

# CONFIG="joint_fusion/config/base_config.yaml"
CONFIG="joint_fusion/config/config_checkpoint_2026-04-07-04-58-17_fold1.yaml"

# make sure to call from root dir
python -m joint_fusion.testing.tester --config=$CONFIG

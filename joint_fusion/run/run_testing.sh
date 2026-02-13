#!/bin/bash

# CONFIG="joint_fusion/config/base_config.yaml"
CONFIG="joint_fusion/config/config_3f_100ep.yaml"

# make sure to call from root dir
python -m joint_fusion.testing.tester --config=$CONFIG

# Multimodal Joint Fusion 

## Training

- Set the working directory to be `multimodal_learning_T1`.
- Provide a yaml configuration file that is similarly structured as `config/base_config.yaml`. Then, within the `run/run_training.sh`, set the`CONFIG` variable to the path of the desired config file.

For running on ALCF systems, modify the `qsub` prefixed files and use `qusb <path_to_qsub_script>`.

# massimal-machine-learning
Machine learning tools for hyperspectral images based on the Massimal research project. 

## Installation
Download / pull the contents of the repo. Navigate to the repo root folder (containing `pyproject.toml`). Create a virtual environment, e.g. one called "massimal_ml", and activate it. Install in "editable" mode via
    
    pip install -e .

This will download the dependencies and install the code as a module (`massimal_machine_learning`) that can be imported.

## Quick start
Create a U-Net model and print a description of it.

    from massimal_machine_learning.hyspec_cnn import unet
    model = unet(
        input_channels=8,
        output_channels=8,
        first_layer_channels=32,
        depth=2,
    )
    model.summary()


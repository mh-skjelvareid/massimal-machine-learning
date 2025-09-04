from massimal_machine_learning.hyspec_cnn import unet

# Create the U-Net model
model = unet(
    input_channels=8,
    output_channels=8,
    first_layer_channels=32,
    depth=2,
)
model.summary()

# import tensorflow as tf

# print(tf.__version__)
# print(tf.keras.__file__)

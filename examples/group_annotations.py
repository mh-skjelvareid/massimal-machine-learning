# from massimal_machine_learning.hyspec_cnn import unet
import json
from importlib.resources import files

from massimal_machine_learning import annotation
from massimal_machine_learning.annotation import class_indices_from_hierarchy, read_hasty_metadata


def load_class_hierarchy():
    # Get the path to the class hierarchy JSON file
    class_hierarchy_path = files("massimal_machine_learning.data").joinpath(
        "massimal_annotation_class_hierarchy.json"
    )
    # Read and parse the JSON file
    with class_hierarchy_path.open("r") as f:
        return json.load(f)


if __name__ == "__main__":
    # Define grouped classes
    grouped_classes = ["Sand", "Bedrock", "Cobble", "Maerl", "Rockweed", "Kelp", "Chorda filum"]
    class_hierarchy = load_class_hierarchy()
    hasty_metadata_path = ""  # To be defined
    class_indices = read_hasty_metadata(hasty_metadata_path)
    grouped_class_indices = annotation.class_indices_from_hierarchy(
        class_hierarchy, class_indices, grouped_classes
    )

    print(grouped_class_indices)

    # Usage
    class_hierarchy = load_class_hierarchy()
    print(class_hierarchy)

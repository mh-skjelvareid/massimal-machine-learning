from massimal_machine_learning.annotation import class_indices_from_hierarchy


def test_class_indices_from_hierarchy():
    class_indices = {
        "Vegetation": 1,
        "Grass": 2,
        "Trees": 3,
        "Oak": 4,
        "Birch": 5,
        "Rock": 6,
        "Bedrock": 7,
        "Cobble": 8,
        "Buildings": 9,
        "Test": 10,
    }
    class_hierarchy = {
        "Vegetation": {
            "Grass": [],
            "Trees": ["Oak", "Birch"],
        },
        "Rock": ["Bedrock", "Cobble"],
        "Buildings": [],
    }
    grouped_class_indices = class_indices_from_hierarchy(
        class_hierarchy, class_indices, ["Grass", "Trees", "Rock"]
    )
    assert grouped_class_indices == {
        "Grass": {2},  # Grass is a leaf node, so it only includes itself
        "Trees": {3, 4, 5},  # Trees includes itself and its children Oak and Birch
        "Rock": {6, 7, 8},  # Rock includes itself and its children Bedrock and Cobble
    }

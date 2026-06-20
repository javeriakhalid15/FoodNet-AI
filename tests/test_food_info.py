import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from food_info import FOOD_INFO, CLASS_NAMES, FOLDER_TO_CLASS


def test_every_class_has_food_info():
    """Every class listed in CLASS_NAMES must have a corresponding
    entry in FOOD_INFO, otherwise the app would crash showing nutrition
    info for that prediction."""
    for cls in CLASS_NAMES:
        assert cls in FOOD_INFO, f"Missing FOOD_INFO entry for class: {cls}"


def test_food_info_has_required_fields():
    """Each FOOD_INFO entry must contain all fields the Streamlit app
    expects to display."""
    required_fields = ["category", "calories", "protein", "carbohydrates", "fat", "fiber", "benefits", "uses"]
    for cls, info in FOOD_INFO.items():
        for field in required_fields:
            assert field in info, f"{cls} is missing field: {field}"


def test_folder_to_class_maps_to_valid_classes():
    """Every value in FOLDER_TO_CLASS must be a real class name,
    otherwise dataset loading would silently mislabel images."""
    for folder, cls in FOLDER_TO_CLASS.items():
        assert cls in CLASS_NAMES, f"FOLDER_TO_CLASS maps '{folder}' to invalid class '{cls}'"


def test_class_names_count():
    """Sanity check: exactly 15 classes as required by the project spec."""
    assert len(CLASS_NAMES) == 15
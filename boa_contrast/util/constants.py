import enum


class IVContrast(enum.IntEnum):
    NON_CONTRAST = 0
    PULMONARY = 1
    ARTERIAL = 2
    VENOUS = 3
    UROGRAPHIC = 4


class Contrast_in_GI(enum.IntEnum):
    NO_CONTRAST_IN_GI_TRACT = 0
    CONTRAST_IN_GI_TRACT = 1


INTERESTING_REGIONS = {
    # Veins/Arteries
    "aorta",
    "pulmonary_artery",
    "inferior_vena_cava",
    "portal_vein_and_splenic_vein",
    # GI tract
    # "esophagus",
    "stomach",
    "duodenum",
    "small_bowel",
    "colon",
    "urinary_bladder",
    # Other organs
    "liver",
    "gallbladder",
    "pancreas",
    "kidney_right",
    "kidney_left",
}

VERTICAL_REGIONS = {"inferior_vena_cava", "aorta", "esophagus"}

ORGANS = {
    # GI tract
    "stomach",
    "duodenum",
    "small_bowel",
    "colon",
    "urinary_bladder",
    # Other organs
    "liver",
    "gallbladder",
    "pancreas",
    "kidney_right",
    "kidney_left",
}

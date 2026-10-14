
# ============================================================
# CAPSICUM CLASS INFORMATION
# 13 Classes - Disease, Pest, Deficiency and Healthy
# File: class_info.py
# ============================================================

CLASS_INFO = {
    "Anthracnose": {
        "type": "Disease",
        "category": "Fungal Disease",
        "common_name": "Anthracnose",
        "scientific_cause": "Colletotrichum species",
        "causal_agent": (
            "Colletotrichum species can cause anthracnose "
            "on Capsicum fruits and other plant parts."
        ),
        "symptoms": [
            "Circular, sunken lesions may appear on fruits.",
            "Lesions may become dark as the disease progresses.",
            "Dark spots may develop on infected plant tissues.",
            "Infected fruit may rot under humid conditions."
        ],
        "cause_and_biology": [
            "Caused by fungal pathogens belonging to Colletotrichum.",
            "Infection can spread through infected seeds, plant debris "
            "and splashing water.",
            "Warm and humid conditions can favor disease development."
        ],
        "management": [
            "Remove severely infected fruits and plant debris.",
            "Avoid overhead irrigation where practical.",
            "Improve airflow and field sanitation.",
            "Use healthy seeds and planting material.",
            "Use only locally registered fungicides according to the label."
        ],
        "precautions": [
            "Inspect fruits regularly for sunken lesions.",
            "Avoid leaving infected fruits in the field.",
            "Confirm uncertain symptoms before treatment."
        ]
    },

    "bacterial spot": {
        "type": "Disease",
        "category": "Bacterial Disease",
        "common_name": "Bacterial Spot",
        "scientific_cause": "Xanthomonas species",
        "causal_agent": (
            "Bacterial pathogens in the Xanthomonas group can cause "
            "bacterial spot in Capsicum."
        ),
        "symptoms": [
            "Small water-soaked spots may appear on leaves.",
            "Spots may become brown or dark with yellow margins.",
            "Severely affected leaves may yellow and fall.",
            "Raised or scabby lesions may appear on fruits."
        ],
        "cause_and_biology": [
            "Caused by pathogenic bacteria in the Xanthomonas group.",
            "Spread can occur through infected seed, transplants, "
            "splashing water and contaminated tools.",
            "Warm, wet weather can favor infection."
        ],
        "management": [
            "Use certified healthy seeds and disease-free transplants.",
            "Remove heavily infected plant debris.",
            "Avoid working among wet plants.",
            "Disinfect tools and equipment appropriately.",
            "Use resistant varieties where available.",
            "Follow local agricultural recommendations for bacterial diseases."
        ],
        "precautions": [
            "Avoid overhead watering where practical.",
            "Do not handle wet plants unnecessarily.",
            "Do not rely on visual symptoms alone for confirmation."
        ]
    },

    "blossom-end rot": {
        "type": "Physiological Disorder",
        "category": "Calcium-Related Disorder",
        "common_name": "Blossom-End Rot",
        "scientific_cause": "Not caused by a pathogen",
        "causal_agent": (
            "A localized calcium supply problem in developing fruit, "
            "often associated with irregular water availability."
        ),
        "symptoms": [
            "A water-soaked spot develops at the blossom end of the fruit.",
            "The affected area becomes brown, dark and sunken.",
            "The lesion may enlarge as the fruit develops.",
            "Secondary organisms may colonize damaged tissue."
        ],
        "cause_and_biology": [
            "This is a physiological disorder, not an infectious disease.",
            "It is associated with insufficient calcium delivery to "
            "developing fruit tissue.",
            "Irregular irrigation, root damage and salinity may contribute."
        ],
        "management": [
            "Maintain consistent soil moisture.",
            "Avoid cycles of severe drying and excessive watering.",
            "Check soil pH and soil nutrient status.",
            "Apply calcium only when assessment indicates it is appropriate.",
            "Avoid excessive nitrogen or potassium fertilization."
        ],
        "precautions": [
            "Do not treat it as a fungal or bacterial infection.",
            "Check irrigation and root conditions.",
            "Soil calcium alone does not always indicate calcium delivery to fruit."
        ]
    },

    "down leaf aphid": {
        "type": "Pest",
        "category": "Insect Pest",
        "common_name": "Aphid Infestation",
        "scientific_cause": "Aphid species",
        "causal_agent": (
            "Sap-feeding aphids may colonize leaf undersides and "
            "tender plant growth."
        ),
        "symptoms": [
            "Small insects may cluster on the undersides of leaves.",
            "Leaves may curl, distort or become yellow.",
            "Sticky honeydew may appear on plant surfaces.",
            "Sooty mold may develop on honeydew.",
            "Some aphids can transmit plant viruses."
        ],
        "cause_and_biology": [
            "Aphids are sap-feeding insects.",
            "Populations can increase rapidly under favorable conditions.",
            "Winged aphids may spread between plants."
        ],
        "management": [
            "Inspect leaf undersides and young shoots.",
            "Remove heavily infested plant parts where practical.",
            "Conserve beneficial insects such as ladybirds.",
            "Manage weeds that may host pests or viruses.",
            "Use locally recommended control measures when necessary."
        ],
        "precautions": [
            "Check new growth regularly.",
            "Avoid unnecessary broad-spectrum insecticide use.",
            "Confirm the insect before selecting a treatment."
        ]
    },

    "fruit thrips": {
        "type": "Pest",
        "category": "Insect Pest",
        "common_name": "Fruit Thrips",
        "scientific_cause": "Thrips species",
        "causal_agent": (
            "Thrips feeding on flowers and developing Capsicum fruits "
            "can cause scarring and surface damage."
        ),
        "symptoms": [
            "Silvery or bronze feeding scars may appear.",
            "Fruit surfaces may become rough or discolored.",
            "Small insects may be visible in flowers or on young fruits.",
            "Severe feeding may deform developing fruits."
        ],
        "cause_and_biology": [
            "Thrips are small insects that feed by rasping plant tissue.",
            "They may hide in flowers, buds and protected plant areas.",
            "Some thrips species can transmit plant viruses."
        ],
        "management": [
            "Inspect flowers and young fruits regularly.",
            "Use suitable monitoring methods, such as sticky traps.",
            "Manage weeds and crop residues appropriately.",
            "Conserve natural enemies.",
            "Use registered pest-control products only when necessary."
        ],
        "precautions": [
            "Inspect flowers as well as leaves.",
            "Do not confuse feeding scars with infectious lesions.",
            "Follow local integrated pest management recommendations."
        ]
    },

    "healthy": {
        "type": "Healthy",
        "category": "Healthy Plant",
        "common_name": "Healthy Capsicum",
        "scientific_cause": "No specific disease or pest identified",
        "causal_agent": "No visible disease or pest cause identified by the classifier.",
        "symptoms": [
            "Leaves generally have normal coloration.",
            "Plant growth appears normal for its growth stage.",
            "No obvious characteristic lesions or pest damage are visible.",
            "Fruits may develop normally when growing conditions are suitable."
        ],
        "cause_and_biology": [
            "The image is classified as healthy by the trained model.",
            "A healthy prediction does not guarantee that the plant "
            "is free from all diseases or pests."
        ],
        "management": [
            "Continue regular crop monitoring.",
            "Maintain balanced nutrition and suitable irrigation.",
            "Use proper field sanitation.",
            "Inspect leaves, stems, roots and fruits periodically."
        ],
        "precautions": [
            "Monitor for new symptoms over time.",
            "Use additional inspection if the plant appears abnormal.",
            "Do not interpret a healthy image prediction as a guarantee."
        ]
    },

    "Larva": {
        "type": "Pest",
        "category": "Larval Insect Pest",
        "common_name": "Insect Larva",
        "scientific_cause": "Unspecified insect larva",
        "causal_agent": (
            "An insect larva may feed on Capsicum leaves, stems, "
            "flowers or fruits, depending on the species."
        ),
        "symptoms": [
            "Chewed or missing leaf tissue may be visible.",
            "Irregular holes may appear on leaves.",
            "Frass or feeding waste may be present.",
            "Some larvae may damage stems or fruits."
        ],
        "cause_and_biology": [
            "Larvae are immature stages of insects.",
            "The type of damage depends on the insect species.",
            "Not every larva found on a plant is a harmful pest."
        ],
        "management": [
            "Inspect plants to identify the larva.",
            "Hand-pick identifiable harmful larvae where practical.",
            "Monitor leaves, stems and fruits for feeding damage.",
            "Conserve beneficial insects.",
            "Use species-appropriate controls only when needed."
        ],
        "precautions": [
            "Do not identify a specific insect species from this label alone.",
            "Check for feeding damage and the actual larva.",
            "Use protective equipment when handling unknown larvae."
        ]
    },

    "Magnesium": {
        "type": "Nutrient Deficiency",
        "category": "Magnesium Deficiency",
        "common_name": "Magnesium Deficiency",
        "scientific_cause": "Insufficient magnesium availability or uptake",
        "causal_agent": (
            "Low magnesium availability or restricted magnesium uptake "
            "may cause deficiency symptoms in Capsicum."
        ),
        "symptoms": [
            "Older leaves may develop yellowing between the veins.",
            "Leaf veins may remain greener than surrounding tissue.",
            "Symptoms may progress to browning or necrosis.",
            "Severe deficiency may reduce plant growth."
        ],
        "cause_and_biology": [
            "Magnesium is essential for chlorophyll and photosynthesis.",
            "Low soil magnesium or nutrient imbalance can contribute.",
            "Excessive potassium or other soil conditions may reduce uptake."
        ],
        "management": [
            "Test soil or plant tissue when possible.",
            "Check soil pH and overall nutrient balance.",
            "Apply an appropriate magnesium source if deficiency is confirmed.",
            "Follow local fertilizer recommendations and application rates."
        ],
        "precautions": [
            "Do not apply excessive magnesium without assessment.",
            "Magnesium symptoms can resemble other nutrient deficiencies.",
            "Confirm suspected deficiency using soil or tissue analysis when possible."
        ]
    },

    "Phytophthora blight": {
        "type": "Disease",
        "category": "Oomycete Disease",
        "common_name": "Phytophthora Blight of Capsicum",
        "scientific_cause": "Phytophthora capsici",
        "causal_agent": (
            "Phytophthora capsici is an oomycete pathogen that can affect "
            "roots, crowns, stems, leaves and fruit."
        ),
        "affected_parts": [
            "Roots",
            "Root Crown",
            "Stems",
            "Leaves",
            "Fruit"
        ],
        "symptoms": [
            "Dark, water-soaked lesions may develop on stems.",
            "Root and crown infection can cause wilting and plant collapse.",
            "Leaves may wilt or develop necrotic areas.",
            "Fruit may develop water-soaked lesions and decay.",
            "Symptoms may progress rapidly under favorable conditions."
        ],
        "favorable_conditions": [
            "Waterlogged or poorly drained soil.",
            "Heavy rainfall or excessive irrigation.",
            "Warm and wet growing conditions.",
            "Movement of contaminated water or soil."
        ],
        "cause_and_biology": [
            "Phytophthora capsici is an oomycete, not a true fungus.",
            "The pathogen can survive in soil and infected plant debris.",
            "Water can spread infectious propagules through a field.",
            "Prolonged soil saturation increases disease risk."
        ],
        "agricultural_impact": [
            "Can cause root, crown and stem damage.",
            "May cause rapid wilting and plant death.",
            "Fruit infection can reduce marketable yield.",
            "Outbreaks can spread through contaminated water."
        ],
        "diagnostic_notes": [
            "Wilting alone is not sufficient to confirm Phytophthora blight.",
            "Inspect roots, crown, stems and fruit for compatible lesions.",
            "Symptoms can resemble other root rots and wilting disorders.",
            "Laboratory confirmation may be needed for a definitive diagnosis."
        ],
        "management": [
            "Improve field drainage and avoid waterlogging.",
            "Use healthy planting material.",
            "Avoid moving contaminated soil or irrigation water.",
            "Remove severely affected plants where appropriate.",
            "Clean tools and equipment after contact with infected soil.",
            "Use resistant or tolerant varieties where available.",
            "Use only locally registered crop-protection products as directed."
        ],
        "ipm": [
            "Inspect plants regularly for wilting and stem or crown lesions.",
            "Monitor low-lying and poorly drained areas.",
            "Use raised beds and suitable drainage where practical.",
            "Avoid excessive irrigation and movement of contaminated soil.",
            "Rotate with suitable non-host crops where recommended."
        ],
        "precautions": [
            "Do not diagnose from wilting alone.",
            "Avoid transporting potentially contaminated soil or water.",
            "Follow local product labels and agricultural recommendations."
        ]
    },

    "powdery mildew": {
        "type": "Disease",
        "category": "Fungal Disease",
        "common_name": "Powdery Mildew",
        "scientific_cause": "Powdery mildew fungi",
        "causal_agent": (
            "Powdery mildew fungi can infect Capsicum leaves and "
            "produce a pale, powder-like growth."
        ),
        "symptoms": [
            "White or pale powdery patches may develop on leaves.",
            "Yellowing may appear around affected areas.",
            "Leaves may dry or drop when infection is severe.",
            "Plant vigor may decline."
        ],
        "cause_and_biology": [
            "Caused by powdery mildew fungi.",
            "Disease development depends on the pathogen and environment.",
            "Some infections can develop without free water on leaf surfaces."
        ],
        "management": [
            "Monitor both upper and lower leaf surfaces.",
            "Improve airflow and avoid excessive canopy density.",
            "Remove heavily affected plant material when practical.",
            "Use resistant varieties where available.",
            "Use registered fungicides according to local guidance."
        ],
        "precautions": [
            "Do not confuse powdery growth with spray residue.",
            "Inspect several leaves before deciding on treatment.",
            "Avoid unnecessary fungicide applications."
        ]
    },

    "snail": {
        "type": "Pest",
        "category": "Mollusc Pest",
        "common_name": "Snail Damage",
        "scientific_cause": "Snail species",
        "causal_agent": (
            "Snails may feed on tender Capsicum leaves, seedlings "
            "and fruits."
        ),
        "symptoms": [
            "Irregular holes may appear on leaves.",
            "Seedlings may be chewed or severely damaged.",
            "Slime trails may be visible on plants or soil.",
            "Damage may be more noticeable in damp conditions."
        ],
        "cause_and_biology": [
            "Snails are molluscs, not insects.",
            "They often shelter in moist, shaded locations.",
            "Activity commonly increases during damp periods."
        ],
        "management": [
            "Remove hiding places and excessive debris near plants.",
            "Manage irrigation to reduce persistent damp shelter areas.",
            "Hand-pick snails where practical.",
            "Protect seedlings using suitable physical barriers.",
            "Use only locally permitted mollusc-control products when needed."
        ],
        "precautions": [
            "Inspect plants during damp periods.",
            "Avoid indiscriminate pesticide use.",
            "Follow product labels if a control product is used."
        ]
    },

    "upperleaf thrips": {
        "type": "Pest",
        "category": "Insect Pest",
        "common_name": "Thrips Feeding Damage",
        "scientific_cause": "Thrips species",
        "causal_agent": (
            "Thrips feeding on leaves and tender plant growth may "
            "cause surface scarring and distortion."
        ),
        "symptoms": [
            "Silvery or pale feeding marks may appear on leaves.",
            "Leaves may become distorted or curled.",
            "Tiny slender insects may be present.",
            "Dark specks of insect waste may appear near feeding sites."
        ],
        "cause_and_biology": [
            "Thrips are small insects that feed on plant tissue.",
            "They can hide in buds, flowers and folded leaves.",
            "Some species may transmit plant viruses."
        ],
        "management": [
            "Inspect young leaves and growing points.",
            "Use sticky traps for monitoring where appropriate.",
            "Manage weeds that may host thrips.",
            "Conserve beneficial insects.",
            "Use locally registered controls when justified."
        ],
        "precautions": [
            "Confirm thrips presence before treatment.",
            "Inspect plants regularly for changing symptoms.",
            "Do not confuse feeding injury with nutrient deficiency."
        ]
    },

    "virus": {
        "type": "Disease",
        "category": "Viral Disease",
        "common_name": "Possible Viral Infection",
        "scientific_cause": "Plant virus; species not identified by this label",
        "causal_agent": (
            "A plant virus may cause mosaic patterns, leaf distortion, "
            "stunting or other symptoms in Capsicum."
        ),
        "symptoms": [
            "Mosaic or mottled leaf coloration may occur.",
            "Leaves may become curled, narrow or distorted.",
            "Plant growth may be stunted.",
            "Fruits may become malformed or unevenly colored."
        ],
        "cause_and_biology": [
            "Symptoms depend on the virus species and plant response.",
            "Some viruses spread through insect vectors such as aphids or thrips.",
            "Others may spread through infected planting material or contact."
        ],
        "management": [
            "Remove and dispose of confirmed infected plants appropriately.",
            "Use healthy seeds and disease-free transplants.",
            "Manage insect vectors using integrated pest management.",
            "Control weeds that may host viruses or vectors.",
            "Disinfect tools when relevant to the suspected virus."
        ],
        "precautions": [
            "A visual prediction cannot identify the exact virus species.",
            "Nutrient disorders and herbicide injury may resemble viral symptoms.",
            "Seek laboratory confirmation when an exact diagnosis is important."
        ]
    }
}


# ============================================================
# CLASS ORDER
# IMPORTANT: Must match the class-index order used during training.
# Verify this order against the training script/checkpoint.
# ============================================================

SUPPORTED_CLASSES = [
    "Anthracnose",
    "bacterial spot",
    "blossom-end rot",
    "down leaf aphid",
    "fruit thrips",
    "healthy",
    "Larva",
    "Magnesium",
    "Phytophthora blight",
    "powdery mildew",
    "snail",
    "upperleaf thrips",
    "virus"
]


# ============================================================
# VALIDATION
# ============================================================

if len(SUPPORTED_CLASSES) != 13:
    raise ValueError(
        f"Expected 13 classes, found {len(SUPPORTED_CLASSES)}."
    )

if len(set(SUPPORTED_CLASSES)) != len(SUPPORTED_CLASSES):
    raise ValueError("Duplicate class names found in SUPPORTED_CLASSES.")

if set(SUPPORTED_CLASSES) != set(CLASS_INFO.keys()):
    missing_info = set(SUPPORTED_CLASSES) - set(CLASS_INFO.keys())
    extra_info = set(CLASS_INFO.keys()) - set(SUPPORTED_CLASSES)

    raise ValueError(
        f"Class information mismatch. "
        f"Missing information: {missing_info}. "
        f"Extra information: {extra_info}."
    )


def get_class_info(class_name: str) -> dict:
    """
    Return information for a predicted class.

    The supplied class name must match a key in CLASS_INFO.
    """
    if class_name not in CLASS_INFO:
        return {
            "type": "Unknown",
            "category": "Unknown",
            "common_name": class_name,
            "scientific_cause": "Unknown",
            "causal_agent": "No information available for this class.",
            "symptoms": [],
            "cause_and_biology": [],
            "management": [],
            "precautions": [
                "Verify the predicted class name and model labels."
            ]
        }

    return CLASS_INFO[class_name]


def get_supported_classes() -> list:
    """Return the configured 13 Capsicum classes."""
    return SUPPORTED_CLASSES.copy()


if __name__ == "__main__":
    print("=" * 60)
    print("CAPSICUM CLASS INFORMATION VALIDATION")
    print("=" * 60)
    print(f"Total classes: {len(SUPPORTED_CLASSES)}")

    for index, class_name in enumerate(SUPPORTED_CLASSES):
        info = CLASS_INFO[class_name]
        print(
            f"{index:02d}. {class_name} "
            f"-> {info['type']} / {info['category']}"
        )

    print("=" * 60)
    print("Validation successful.")

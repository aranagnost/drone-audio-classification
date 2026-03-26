import json
from pathlib import Path

# Path to your metadata
metadata_path = Path("datasets/Drone_Audio_Dataset/metadata.json")

# List of files to remove (from your warning output)
to_remove = [
    "0bV7F4sbA9Q_8_motors_000.wav",
    "0bV7F4sbA9Q_8_motors_002.wav",
    "0bV7F4sbA9Q_8_motors_003.wav",
    "0bV7F4sbA9Q_8_motors_004.wav",
    "0bV7F4sbA9Q_8_motors_005.wav",
    "0bV7F4sbA9Q_8_motors_006.wav",
    "0bV7F4sbA9Q_8_motors_007.wav",
    "0bV7F4sbA9Q_8_motors_008.wav",
    "0bV7F4sbA9Q_8_motors_009.wav",
    "0bV7F4sbA9Q_8_motors_010.wav",
    "0bV7F4sbA9Q_8_motors_011.wav",
    "0bV7F4sbA9Q_8_motors_012.wav",
    "0bV7F4sbA9Q_8_motors_013.wav",
    "0bV7F4sbA9Q_8_motors_014.wav",
    "0bV7F4sbA9Q_8_motors_015.wav",
    "0bV7F4sbA9Q_8_motors_016.wav",
    "0bV7F4sbA9Q_8_motors_017.wav",
    "0bV7F4sbA9Q_8_motors_125.wav",
    "0bV7F4sbA9Q_8_motors_157.wav",
    "0bV7F4sbA9Q_8_motors_162.wav",
    "0bV7F4sbA9Q_8_motors_163.wav",
    "0bV7F4sbA9Q_8_motors_164.wav",
    "0bV7F4sbA9Q_8_motors_165.wav",
    "0bV7F4sbA9Q_8_motors_166.wav",
    "0bV7F4sbA9Q_8_motors_167.wav",
    "0bV7F4sbA9Q_8_motors_168.wav",
    "0bV7F4sbA9Q_8_motors_169.wav",
    "0bV7F4sbA9Q_8_motors_170.wav",
    "0bV7F4sbA9Q_8_motors_171.wav",
    "0bV7F4sbA9Q_8_motors_172.wav",
    "0bV7F4sbA9Q_8_motors_173.wav",
    "0bV7F4sbA9Q_8_motors_174.wav",
    "7U-n8dkRFgU_8_motors_000.wav",
    "7U-n8dkRFgU_8_motors_001.wav",
    "7U-n8dkRFgU_8_motors_002.wav",
    "7U-n8dkRFgU_8_motors_003.wav",
    "7U-n8dkRFgU_8_motors_004.wav",
    "7U-n8dkRFgU_8_motors_005.wav",
    "B7cA_7NKwFA_6_motors_000.wav",
    "B7cA_7NKwFA_6_motors_016.wav",
    "B7cA_7NKwFA_6_motors_017.wav",
    "DLIhJMAYKXc_6_motors_000.wav",
    "DLIhJMAYKXc_6_motors_001.wav",
    "DLIhJMAYKXc_6_motors_008.wav",
    "DLIhJMAYKXc_6_motors_009.wav",
    "DLIhJMAYKXc_6_motors_010.wav",
    "DLIhJMAYKXc_6_motors_011.wav",
    "DLIhJMAYKXc_6_motors_012.wav",
    "DLIhJMAYKXc_6_motors_016.wav",
    "DLIhJMAYKXc_6_motors_017.wav",
    "iSG2J2NRgVo_8_motors_014.wav"
]

# Load existing metadata
with open(metadata_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Filter out entries
data_filtered = [entry for entry in data if entry["filename"] not in to_remove]

# Save back
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(data_filtered, f, indent=2)

print(f"Removed {len(data) - len(data_filtered)} entries from metadata.json")

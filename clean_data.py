import re

# File paths
input_file = "webmd.txt"
output_file = "data/webmd_clean.txt"

# Read text from input file
with open(input_file, "r", encoding="utf-8") as file:
    text = file.read()

# Remove img and svg links
text = re.sub(r"!\[\]\(.*?\.svg\)", "", text)

# Remove https links
text = re.sub(r"\[.*?\]\(https?://.*?\)", "", text)

# Save cleaned text to output file
with open(output_file, "w", encoding="utf-8") as file:
    file.write(text.strip())

print(f"Cleaned text saved to {output_file}.")

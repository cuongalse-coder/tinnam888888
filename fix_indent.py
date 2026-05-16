import sys

filename = "streamlit_app.py"
with open(filename, "r", encoding="utf-8") as f:
    lines = f.readlines()

for i in range(856, min(1064, len(lines))):
    if lines[i].strip() != "":
        lines[i] = "    " + lines[i]

with open(filename, "w", encoding="utf-8") as f:
    f.writelines(lines)

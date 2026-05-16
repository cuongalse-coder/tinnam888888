filename = "streamlit_app.py"
with open(filename, "r", encoding="utf-8") as f:
    lines = f.readlines()

for i in range(531, 560): # 532 is index 531
    lines[i] = "    " + lines[i]

with open(filename, "w", encoding="utf-8") as f:
    f.writelines(lines)

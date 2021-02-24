import os

directory='./'

for filename in os.listdir(directory):
    if filename.endswith(".png"):
        file = os.path.join(directory, filename)
        file2 = os.path.join(directory, "trimmed/"+filename)
        cmd = f'convert "{file}" -trim "{file2}"'
        print(cmd)
        print(os.system(cmd))

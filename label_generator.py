import os

img_path = os.path.join(os.path.dirname(__file__), "Data", "food-101", "meta", "labels.txt")
result = []
with open(img_path, 'r') as file:
    for line in file:
        line = line.strip()
        line = line.replace(" ", "_")
        result.append(line.lower())
print(result)
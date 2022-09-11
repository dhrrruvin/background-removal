from data import process_data

image, mask = process_data("dataset/train/", "images", "labels")

print(image)
print("-----------------------------------------------------")
print(mask)
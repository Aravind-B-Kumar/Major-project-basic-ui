import torch

path = r"D:\MajorProject\basic-ui\models\best_resnet18_species.pth"

# Load checkpoint
ckpt = torch.load(path, map_location="cpu")
print("Loaded checkpoint type:", type(ckpt))

# Function to search for class labels in nested dicts
def find_class_labels(d, prefix=""):
    for k, v in d.items():
        name = prefix + k

        # Keys that commonly store classes
        if any(tag in k.lower() for tag in ["class", "label", "category", "names"]):
            print(f"\nPossible class info at: {name}")
            print(v)

        # Search inside nested dictionaries
        if isinstance(v, dict):
            find_class_labels(v, prefix=name + ".")

# If checkpoint is a dict (most model files)
if isinstance(ckpt, dict):
    print("\nSearching checkpoint for class labels...")
    find_class_labels(ckpt)
else:
    print("Checkpoint is not a dict. Cannot inspect keys.")

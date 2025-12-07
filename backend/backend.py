import io
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models

app = FastAPI()

# CORS (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# 1. Load Model
# -------------------------
MODEL_PATH = r"models\newtrial.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example: use ResNet18 (replace with your actual model)
model = models.resnet18(pretrained=False)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
class_names = checkpoint['class_names']
NUM_CLASSES = len(class_names)  # Your number of classes
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# Load state dict

model.to(device)
model.eval()

# -------------------------
# 2. Define Preprocessing
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------
# 3. Prediction Endpoint
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        class_name = class_names[predicted.item()]

    return {"class_name": class_name}

import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import io

# ===== Классы активности =====
class_names = ['calling', 'clapping', 'cycling', 'dancing', 'drinking',
               'eating', 'fighting', 'hugging', 'laughing', 'listening_to_music',
               'running', 'sitting', 'sleeping', 'texting', 'using_laptop']

# ===== Настройки модели =====
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load("activity_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# ===== Преобразования изображения =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ===== Инференс одной картинки =====
def predict(image, model):
    image = image.convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    return class_names[predicted.item()], confidence.item()

# ===== Streamlit UI =====
st.title("🧍 Human Activity Recognition (Image)")
st.write("Upload a photo and I’ll tell you what the person is doing.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        model = load_model()
        label, confidence = predict(image, model)
        st.success(f"**Prediction:** {label}")
        st.info(f"Confidence: {confidence * 100:.2f}%")
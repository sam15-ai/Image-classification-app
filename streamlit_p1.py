import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

#UI design
st.markdown(
    """
    <style>
    /* Change background color */
    .stApp {
        background-color: #f0f8ff;  
    }

    /* Center align and resize title */
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold; 
    }

    /* Subtitle styling */
    .subtitle {
        text-align: center;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='title'>🖼️ Image Classification using CIFAR-10</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload an image and the model will predict its class.</p>", unsafe_allow_html=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Resize((224, 224)),   
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# CIFAR-10 classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']


@st.cache_resource  # cache so it loads only once
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load("resnet18_cifar10.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_class = classes[probabilities.argmax().item()]
        confidence = probabilities.max().item() * 100

    # Show results
    st.subheader(f"✅ Prediction: **{predicted_class}**")
    st.write(f"Confidence: {confidence:.2f}%")

    # Show probability bar chart
    st.bar_chart({classes[i]: float(probabilities[i]) for i in range(len(classes))})

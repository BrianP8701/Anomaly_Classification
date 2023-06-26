import torch
from PIL import Image
from torchvision import transforms
from model_repository import model_dict  # Import the model_dict

# You may need to adjust the input size based on the model you are using
# For example, EfficientNet-B0 uses 224, but other versions may use larger sizes
input_size = 224

preprocess = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(model_path, model_name):
    # Verify the requested model name is defined in the dictionary
    if model_name not in model_dict:
        raise ValueError(f"Invalid model name {model_name}. Available models are {list(model_dict.keys())}")

    # Initialize the model architecture
    model_func = model_dict[model_name]
    model = model_func(num_classes=3)

    # Load the state dictionary
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model

def infer_image(model, image_path):
    image = Image.open(image_path)
    image = preprocess(image)
    image = image.unsqueeze(0)  # create a mini-batch as expected by the model
    
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1)

    return predicted_class.item()


# Load model and run inference
model_path = 'models/models3/res18_gmms6_224_transfer.pt'
image_path = 'datasets/gmms6_224/val/over/frame3.jpg'
model = load_model(model_path, 'resnet18')
predicted_class = infer_image(model, image_path)
print(predicted_class)

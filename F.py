from flask import Flask, render_template, request
from PIL import Image, ImageEnhance
from io import BytesIO
import torch
from torchvision import transforms
import seaborn as sns
import matplotlib.pyplot as plt

app = Flask(__name__)

# HTML templates are stored in the "templates" folder

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle image upload and processing
        file = request.files['file']
        
        if file:
            original_image, normalized_image, brightness_comparison_plot = process_image(file)
            return render_template("result.html",
                                   original_image=original_image,
                                   normalized_image=normalized_image,
                                   brightness_comparison_plot=brightness_comparison_plot)

    return render_template("index.html")

def process_image(file):
    # Read the uploaded image
    image = Image.open(file)
    
    # Convert PIL Image to PyTorch Tensor
    transform = transforms.Compose([transforms.ToTensor()])
    tensor_image = transform(image).unsqueeze(0)
    
    # Calculate mean of the image tensor
    original_mean = tensor_image.mean().item()

    # Normalize brightness using PyTorch
    brightness_factor = 0.5  # You can adjust this factor as needed
    normalized_image_tensor = transforms.functional.adjust_brightness(tensor_image, brightness_factor)
    
    # Convert the normalized tensor back to a PIL Image
    normalized_image = transforms.ToPILImage()(normalized_image_tensor.squeeze(0))
    
    # Plot mean brightness before and after normalization using Seaborn
    normalized_mean = normalized_image_tensor.mean().item()
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=["Original", "Normalized"],
                y=[original_mean, normalized_mean],
                palette=["blue", "orange"])
    plt.title("Mean Brightness Comparison")
    plt.xlabel("Image Type")
    plt.ylabel("Mean Brightness")
    brightness_comparison_plot = get_image_from_plot()
    
    return get_image_from_pil(image), get_image_from_pil(normalized_image), brightness_comparison_plot

def get_image_from_pil(pil_image):
    img_io = BytesIO()
    pil_image.save(img_io, 'PNG')
    img_io.seek(0)
    return img_io.getvalue()

def get_image_from_plot():
    img_io = BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    plt.close()
    return img_io.getvalue()

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=8080)

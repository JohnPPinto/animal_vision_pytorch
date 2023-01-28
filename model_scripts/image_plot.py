"""
Contins code to predict and plot an image
"""
from typing import List, Tuple
from PIL import Image
import torch
import torchvision
import matplotlib.pyplot as plt

# defining device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# creating a function for predicting and plotting
def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str],
                        image_size: Tuple[int, int],
                        transform: torchvision.transforms = None,
                        device: torch.device = device):
    """
    Predict and plot an image
    Args:
      model: A model to perform prediction.
      image_path: A path string of the image location.
      class_names: A list of all the classes names.
      image_size: A tuple with image size in shape of (height, width).
      transform: Torchvision transforms compose class to transform the data,
      device: A device either 'cuda' or 'cpu'.
    """
    # open image
    img = Image.open(image_path)
    # create transformation
    if transform is not None:
        img_transform = transform
    else:
        img_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])
    # predict image
    model.to(device)
    model.eval()
    with torch.inference_mode():
        img_transform = img_transform(img).unsqueeze(dim=0)
        img_pred = model(img_transform.to(device))
    img_pred_probs = torch.softmax(img_pred, dim=1)
    img_pred_label = torch.argmax(img_pred_probs, dim=1)
    # plot image
    plt.figure()
    plt.imshow(img)
    plt.title(f'Pred: {class_names[img_pred_label]} | Prob: {img_pred_probs.max():.3f}')
    plt.axis(False);

import torch
from torch import nn
import numpy as np
from PIL import Image

# recreate model structure
model = nn.Sequential(
    nn.Linear(in_features=28*28, out_features=64, bias=True),
    nn.ReLU(),
    nn.Linear(in_features=64, out_features=64, bias=True),
    nn.ReLU(),
    nn.Linear(in_features=64, out_features=10, bias=True)
)

# load model from model.pth file
model.load_state_dict(torch.load("model.pth"))
model.eval()

# load single test image
im = np.array(Image.open('test.jpg').convert('L'))
im = im.flatten()
data = [0.0 for i in range(28*28)] 

# create tensor from image
for i in range(28*28):
    data[i] = float(round(im[i]/255, 4))
input = torch.tensor(data)

# pass tensor to model
with torch.no_grad():
    pred = model(input)
    print(f'Model prediction: {torch.argmax(pred).item()}')
    print(f'Output nodes: {pred}')
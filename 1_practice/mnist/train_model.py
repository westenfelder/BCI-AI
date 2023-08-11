import torch
from torch import nn
from torch import optim
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

# Download MNIST dataset
train = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train, batch_size=50)

#define model 784 -> 64 -> 10 nodes
model = nn.Sequential(
    nn.Linear(in_features=28*28, out_features=64, bias=True),
    nn.ReLU(),
    nn.Linear(in_features=64, out_features=64, bias=True),
    nn.ReLU(),
    nn.Linear(in_features=64, out_features=10, bias=True)
)

#define optimizer
optimizer = optim.SGD(params=model.parameters(), lr=.1)

#define loss function
loss = nn.CrossEntropyLoss()

#training and validation loops
number_epochs = 10
for epoch in range(number_epochs):
    for batch in train_loader:
        # load batch of images and corresponding numbers
        images, numbers = batch
        batch_size = images.size(0)
        images = images.view(batch_size, -1)

        # Run batch
        forward = model(images)
        # calculate the loss
        loss_function = loss(forward, numbers)
        # clean gradients
        model.zero_grad()
        # accumulate the partial derivatives
        loss_function.backward()
        # step in opposite direction of gradients
        optimizer.step()

    print(f'Epoch {epoch+1} complete')

#save model
torch.save(model.state_dict(), "model.pth")
print("Model saved to model.pth")
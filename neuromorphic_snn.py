"""
Neuromorphic Spiking Neural Network (SNN)
Single-file implementation: model, training, evaluation

Uses:
- Leaky Integrate-and-Fire (LIF) neurons
- Surrogate gradient for training
- Simple classification task (MNIST)

Author: Your Name
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# ----------------------------
# Hyperparameters
# ----------------------------
BATCH_SIZE = 64
TIME_STEPS = 25
LR = 1e-3
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LIF neuron parameters
THRESHOLD = 1.0
DECAY = 0.25


# ----------------------------
# Surrogate Gradient Function
# ----------------------------
class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad = grad_output.clone()
        # surrogate gradient (fast sigmoid)
        return grad / (1.0 + torch.abs(input))**2


spike_fn = SpikeFunction.apply


# ----------------------------
# LIF Neuron Layer
# ----------------------------
class LIFLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x, mem):
        mem = mem * DECAY + self.fc(x)
        spike = spike_fn(mem - THRESHOLD)
        mem = mem * (1 - spike)  # reset
        return spike, mem


# ----------------------------
# Spiking Neural Network
# ----------------------------
class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = LIFLayer(28 * 28, 256)
        self.fc2 = LIFLayer(256, 10)

    def forward(self, x):
        batch_size = x.size(0)

        mem1 = torch.zeros(batch_size, 256).to(DEVICE)
        mem2 = torch.zeros(batch_size, 10).to(DEVICE)

        spk2_rec = []

        for _ in range(TIME_STEPS):
            cur = x.view(batch_size, -1)

            spk1, mem1 = self.fc1(cur, mem1)
            spk2, mem2 = self.fc2(spk1, mem2)

            spk2_rec.append(spk2)

        out = torch.stack(spk2_rec).sum(0)  # spike count
        return out


# ----------------------------
# Data
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + 0.01 * torch.rand_like(x))  # noise encoding
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=True, download=True, transform=transform),
    batch_size=BATCH_SIZE, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=False, transform=transform),
    batch_size=BATCH_SIZE, shuffle=False
)


# ----------------------------
# Training
# ----------------------------
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ----------------------------
# Evaluation
# ----------------------------
def test(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)

            output = model(data)
            pred = output.argmax(dim=1)

            correct += (pred == target).sum().item()
            total += target.size(0)

    return correct / total


# ----------------------------
# Main
# ----------------------------
def main():
    model = SNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        loss = train(model, train_loader, optimizer, criterion)
        acc = test(model, test_loader)

        print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()

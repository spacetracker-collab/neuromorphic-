"""
Neuromorphic SNN + Integrated Information (Phi-like proxy)

- Spiking Neural Network (LIF neurons)
- Surrogate gradient training
- Approximate Integrated Information measure

Author: Your Name
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# ----------------------------
# Device
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Hyperparameters
# ----------------------------
BATCH_SIZE = 64
TIME_STEPS = 20
LR = 1e-3
EPOCHS = 3

THRESHOLD = 1.0
DECAY = 0.25

# ----------------------------
# Surrogate Gradient
# ----------------------------
class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1.0 + torch.abs(input))**2

spike_fn = SpikeFunction.apply

# ----------------------------
# LIF Layer
# ----------------------------
class LIFLayer(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f)

    def forward(self, x, mem):
        mem = mem * DECAY + self.fc(x)
        spike = spike_fn(mem - THRESHOLD)
        mem = mem * (1 - spike)
        return spike, mem

# ----------------------------
# SNN Model
# ----------------------------
class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = LIFLayer(28*28, 128)
        self.fc2 = LIFLayer(128, 10)

    def forward(self, x):
        batch_size = x.size(0)

        mem1 = torch.zeros(batch_size, 128).to(DEVICE)
        mem2 = torch.zeros(batch_size, 10).to(DEVICE)

        spikes_layer1 = []
        spikes_layer2 = []

        for _ in range(TIME_STEPS):
            cur = x.view(batch_size, -1)

            spk1, mem1 = self.fc1(cur, mem1)
            spk2, mem2 = self.fc2(spk1, mem2)

            spikes_layer1.append(spk1)
            spikes_layer2.append(spk2)

        spk1_rec = torch.stack(spikes_layer1)  # [T, B, N]
        spk2_rec = torch.stack(spikes_layer2)

        output = spk2_rec.sum(0)
        return output, spk1_rec

# ----------------------------
# Integrated Information Proxy
# ----------------------------
def compute_phi(spike_tensor):
    """
    Approximate Phi:
    Measures how much joint activity differs from independent activity

    spike_tensor: [T, B, N]
    """
    T, B, N = spike_tensor.shape

    # Flatten time and batch
    data = spike_tensor.reshape(T * B, N)

    # Mean activity per neuron
    mean_activity = data.mean(dim=0)

    # Covariance matrix
    cov = torch.cov(data.T)

    # Total system variance
    total_var = torch.trace(cov)

    # Independent variance (sum of individual variances)
    independent_var = torch.sum(torch.diag(cov))

    # Phi proxy = integration (difference)
    phi = total_var - independent_var

    return phi.item()

# ----------------------------
# Data
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + 0.01 * torch.rand_like(x))
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
# Train
# ----------------------------
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    phi_values = []

    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        output, spikes = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        phi = compute_phi(spikes.detach().cpu())
        phi_values.append(phi)

        total_loss += loss.item()

    avg_phi = sum(phi_values) / len(phi_values)
    return total_loss / len(loader), avg_phi

# ----------------------------
# Test
# ----------------------------
def test(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)

            output, _ = model(data)
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
        loss, phi = train(model, train_loader, optimizer, criterion)
        acc = test(model, test_loader)

        print(f"Epoch {epoch+1}")
        print(f"Loss: {loss:.4f} | Accuracy: {acc:.4f} | Phi (integration): {phi:.6f}")
        print("-" * 50)

if __name__ == "__main__":
    main()

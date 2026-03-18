"""
Global Workspace Theory (GWT) + Neuromorphic SNN

- Multiple spiking modules
- Attention-based competition
- Global broadcast mechanism

Author: Your Name
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Hyperparameters
# ----------------------------
BATCH_SIZE = 64
TIME_STEPS = 20
LR = 1e-3
EPOCHS = 3

NUM_MODULES = 4
HIDDEN = 128

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
# Module (Unconscious Processor)
# ----------------------------
class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = LIFLayer(28*28 + HIDDEN, HIDDEN)

    def forward(self, x, workspace, mem):
        # combine input + global broadcast
        combined = torch.cat([x, workspace], dim=1)
        spike, mem = self.layer(combined, mem)
        return spike, mem

# ----------------------------
# GWT Network
# ----------------------------
class GWT_SNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.modules = nn.ModuleList([Module() for _ in range(NUM_MODULES)])

        # attention to select "winner"
        self.attention = nn.Linear(HIDDEN, 1)

        # output layer
        self.output = nn.Linear(HIDDEN, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)

        # initialize memories
        mems = [torch.zeros(batch_size, HIDDEN).to(DEVICE) for _ in range(NUM_MODULES)]
        workspace = torch.zeros(batch_size, HIDDEN).to(DEVICE)

        workspace_trace = []

        for _ in range(TIME_STEPS):

            module_outputs = []

            # each module processes in parallel
            for i, module in enumerate(self.modules):
                spk, mems[i] = module(x_flat, workspace, mems[i])
                module_outputs.append(spk)

            module_outputs = torch.stack(module_outputs)  # [M, B, H]

            # ----------------------------
            # Competition (Attention)
            # ----------------------------
            scores = []
            for m in module_outputs:
                scores.append(self.attention(m))

            scores = torch.stack(scores)  # [M, B, 1]
            weights = torch.softmax(scores, dim=0)

            # ----------------------------
            # Global Workspace (Broadcast)
            # ----------------------------
            workspace = (weights * module_outputs).sum(0)

            workspace_trace.append(workspace)

        # integrate over time
        workspace_sum = torch.stack(workspace_trace).mean(0)

        out = self.output(workspace_sum)
        return out, workspace_sum

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

    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        output, workspace = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

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
    model = GWT_SNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        loss = train(model, train_loader, optimizer, criterion)
        acc = test(model, test_loader)

        print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()

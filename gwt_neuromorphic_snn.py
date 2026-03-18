# ==========================================
# GWT + Neuromorphic SNN (DEBUGGED VERSION)
# ==========================================

!pip install -q torch torchvision

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# ----------------------------
# Device
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

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
# Module
# ----------------------------
class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = LIFLayer(28*28 + HIDDEN, HIDDEN)

    def forward(self, x, workspace, mem):
        combined = torch.cat([x, workspace], dim=1)
        spike, mem = self.layer(combined, mem)
        return spike, mem

# ----------------------------
# GWT Model
# ----------------------------
class GWT_SNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.module_list = nn.ModuleList([Module() for _ in range(NUM_MODULES)])
        self.attention = nn.Linear(HIDDEN, 1)
        self.output = nn.Linear(HIDDEN, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)

        # initialize states
        mems = [torch.zeros(batch_size, HIDDEN).to(DEVICE) for _ in range(NUM_MODULES)]
        workspace = torch.zeros(batch_size, HIDDEN).to(DEVICE)

        workspace_trace = []

        for t in range(TIME_STEPS):

            module_outputs = []

            # process modules
            for i in range(NUM_MODULES):
                spk, mems[i] = self.module_list[i](x_flat, workspace, mems[i])
                module_outputs.append(spk)

            # stack safely → [M, B, H]
            module_outputs = torch.stack(module_outputs, dim=0)

            # ----------------------------
            # Attention (Competition)
            # ----------------------------
            scores = torch.stack(
                [self.attention(module_outputs[i]) for i in range(NUM_MODULES)],
                dim=0
            )  # [M, B, 1]

            weights = torch.softmax(scores, dim=0)

            # ----------------------------
            # Global Workspace
            # ----------------------------
            workspace = torch.sum(weights * module_outputs, dim=0)

            workspace_trace.append(workspace)

        # temporal integration
        workspace_sum = torch.stack(workspace_trace, dim=0).mean(dim=0)

        out = self.output(workspace_sum)
        return out

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
        output = model(data)

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

            output = model(data)
            pred = output.argmax(dim=1)

            correct += (pred == target).sum().item()
            total += target.size(0)

    return correct / total

# ----------------------------
# Run
# ----------------------------
model = GWT_SNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    loss = train(model, train_loader, optimizer, criterion)
    acc = test(model, test_loader)

    print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")

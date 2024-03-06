# %%
import numpy as np
import pathlib

# data_path = pathlib.Path("../data/2024-03-05_softballs_idx11_augmented_pose/evaled_grasp_config_dicts/ddg-ycb_054_softball_0_0350.npy")
data_path = pathlib.Path("../data/2024-03-05_softballs_idx11_augmented_pose_HALTON/evaled_grasp_config_dicts/ddg-ycb_054_softball_0_0350.npy")
assert data_path.exists()

# %%
data_dict = np.load(data_path, allow_pickle=True).item()

# %%
print(f"data_dict.keys() = {data_dict.keys()}")

import matplotlib.pyplot as plt

# %%
passed_sim = data_dict["passed_simulation"]
passed_penetration = data_dict["passed_new_penetration_test"]
passed_eval = data_dict["passed_eval"]

trans = data_dict["trans"]
rot = data_dict["rot"]

N = passed_sim.shape[0]
print(f"N = {N}")
assert passed_sim.shape == (N,)
assert passed_penetration.shape == (N,)
assert passed_eval.shape == (N,)
assert trans.shape == (N, 3)
assert rot.shape == (N, 3, 3)

# %%
plt.hist(passed_eval)
plt.title("passed_eval")

# %%
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

class GraspDataset(Dataset):
    def __init__(self, trans: np.ndarray, rot: np.ndarray, passed_sim: np.ndarray, passed_penetration: np.ndarray, passed_eval: np.ndarray) -> None:
        self.trans = torch.from_numpy(trans).float()
        self.rot = torch.from_numpy(rot).float()
        self.passed_sim = torch.from_numpy(passed_sim).float()
        self.passed_penetration = torch.from_numpy(passed_penetration).float()
        self.passed_eval = torch.from_numpy(passed_eval).float()

        N = len(self)
        assert self.trans.shape == (N, 3)
        assert self.rot.shape == (N, 3, 3)
        assert self.passed_sim.shape == (N,)
        assert self.passed_penetration.shape == (N,)
        assert self.passed_eval.shape == (N,)

    def __len__(self) -> int:
        return self.trans.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        N = len(self)
        inputs = torch.cat([
            self.trans[idx],
            self.rot[idx].flatten(),
        ],
        dim=0)
        assert inputs.shape == (12,)
        labels = torch.tensor(self.passed_eval[idx]).float().reshape(1)
        assert labels.shape == (1,)
        return inputs, labels

# %%
import torch.nn as nn
import torch.nn.functional as F

class BasicMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        input_size = 12
        output_size = 1
        hidden_layers = [1024] * 5
        self.fcs = nn.ModuleList()

        sizes = [input_size] + hidden_layers + [output_size]
        for layer_size, next_layer_size in zip(sizes[:-1], sizes[1:]):
            self.fcs.append(nn.Linear(layer_size, next_layer_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fc in self.fcs[:-1]:
            x = F.relu(fc(x))
        x = self.fcs[-1](x)
        return x


# %%
from torch.optim import Adam

train_size = int(0.8 * N)
val_size = N - train_size
dataset = GraspDataset(
    trans=trans,
    rot=rot,
    passed_sim=passed_sim,
    passed_penetration=passed_penetration,
    passed_eval=passed_eval,
)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BasicMLP().to(device)
optimizer = Adam(model.parameters())
loss_fn = nn.L1Loss(reduction="none")

def train_epoch(model: BasicMLP, dataloader: DataLoader, loss_fn: nn.Module, optimizer: Adam) -> float:
    model.train()
    total_loss = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        assert outputs.shape == labels.shape
        losses = loss_fn(outputs, labels)
        mean_loss = losses.mean()
        mean_loss.backward()
        optimizer.step()
        total_loss += mean_loss.item()
    return total_loss / len(dataloader)

def eval_model(model: BasicMLP, dataloader: DataLoader, loss_fn: nn.Module) -> float:
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            assert outputs.shape == labels.shape
            losses = loss_fn(outputs, labels)
            mean_loss = losses.mean()
            total_loss += mean_loss.item()
    return total_loss / len(dataloader)

# %%
N_EPOCHS = 1000
train_losses = []
val_losses = []
for epoch in range(N_EPOCHS):
    train_loss = train_epoch(model=model, dataloader=train_loader, loss_fn=loss_fn, optimizer=optimizer)
    val_loss = eval_model(model=model, dataloader=val_loader, loss_fn=loss_fn)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"epoch = {epoch}, train_loss = {train_loss}, val_loss = {val_loss}")

# %%
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.legend()
plt.show()

# %%
def plot_predictions_vs_actual(dataloader: DataLoader, model: nn.Module, title: str):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.view(-1).tolist())
            actuals.extend(labels.view(-1).tolist())
            
    plt.scatter(actuals, predictions, alpha=0.3)
    plt.xlabel('Actual Label')
    plt.ylabel('Predicted Label')
    plt.title(title)
    plt.plot([0, 1], [0, 1], 'red', lw=2)  # Diagonal line for reference
    plt.show()

# Now call the plotting function for both train and val sets
print("Training Set Predictions vs Actual")
plot_predictions_vs_actual(train_loader, model, "Training Set Predictions vs Actual")

print("Validation Set Predictions vs Actual")
plot_predictions_vs_actual(val_loader, model, "Validation Set Predictions vs Actual")

# %%

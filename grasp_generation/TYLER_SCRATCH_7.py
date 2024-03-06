# %%
import numpy as np
import pathlib

# data_path = pathlib.Path("../data/2024-03-05_softballs_idx11_augmented_pose/evaled_grasp_config_dicts/ddg-ycb_054_softball_0_0350.npy")
# data_path = pathlib.Path("../data/2024-03-05_softballs_idx11_augmented_pose_HALTON/evaled_grasp_config_dicts/ddg-ycb_054_softball_0_0350.npy")
data_path = pathlib.Path("../data/PROBE_1_2024-02-07_softball_0-051_5random/evaled_grasp_config_dicts/ddg-ycb_054_softball_0_0510.npy")
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

# N = passed_sim.shape[0]
N = 10000
inds = np.random.choice(np.arange(passed_sim.shape[0]), size=N, replace=False)
print(f"N = {N}")
passed_sim = passed_sim[inds]
passed_penetration = passed_penetration[inds]
passed_eval = passed_eval[inds]
trans = trans[inds]
rot = rot[inds]

assert passed_sim.shape == (N,)
assert passed_penetration.shape == (N,)
assert passed_eval.shape == (N,)
assert trans.shape == (N, 3)
assert rot.shape == (N, 3, 3)

# %%
plt.hist(passed_eval)
plt.title("passed_eval")

# %%
plt.hist(passed_sim)
plt.title("passed_sim")

# %%
plt.scatter(trans[:, 0], trans[:, 1], s=1, c=passed_sim)
plt.title("Passed Sim")

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
        # labels = torch.tensor(self.passed_eval[idx]).float().reshape(1)
        labels = torch.tensor(self.passed_sim[idx]).float().reshape(1)
        labels = torch.concat(
            [1 - labels, labels], dim=-1
        )
        assert labels.shape == (2,)
        return inputs, labels

# %%
import torch.nn as nn
import torch.nn.functional as F

class BasicMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        input_size = 12
        output_size = 2
        hidden_layers = [2048] * 6
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

def assert_equals(a, b):
    assert a == b, f"{a} != {b}"

class WeightedSoftmaxL1(nn.Module):
    def __init__(
        self,
        unique_label_weights: torch.Tensor,
        unique_labels: torch.Tensor,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.unique_label_weights = unique_label_weights
        self.unique_labels = unique_labels

        (N,) = self.unique_labels.shape
        (N2,) = self.unique_label_weights.shape
        assert_equals(N, N2)

        self.l1_loss = nn.L1Loss(*args, **kwargs)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = input.shape[0]
        assert_equals(input.shape, (batch_size, 2))
        assert_equals(target.shape, (batch_size, 2))

        # Compute the raw L1 losses
        input = torch.softmax(input, dim=-1)
        l1_losses = self.l1_loss(input, target).mean(dim=-1)

        assert_equals(l1_losses.shape, (batch_size,))

        # Find the closest label for each target and its index
        target_indices = (
            (self.unique_labels.unsqueeze(0) - target[:, 1:2]).abs().argmin(dim=1)
        )

        # Gather the weights for each sample
        weights = self.unique_label_weights[target_indices]

        # Weight the L1 losses
        weighted_losses = weights * l1_losses

        assert len(weighted_losses.shape) == 1
        return weighted_losses



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
all_loader = DataLoader(dataset, batch_size=64, shuffle=False)

# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BasicMLP().to(device)
optimizer = Adam(model.parameters())
# loss_fn = nn.L1Loss(reduction="none")
loss_fn = nn.MSELoss(reduction="none")
def get_class_from_success_rate(
    success_rates: np.ndarray, unique_classes: np.ndarray
) -> int:
    # [0.5, 0.2, 0.1, 0.5, 0.0, 0.0, ...], [0, 0.1, 0.2, ...] => [5, 2, 1, 5, 0, 0, ...]
    assert unique_classes.ndim == 1
    errors = np.absolute(unique_classes[None, :] - success_rates[:, None])
    return errors.argmin(axis=-1)
from sklearn.utils.class_weight import compute_class_weight
def _compute_class_weight(
    success_rates: np.ndarray, unique_classes: np.ndarray
) -> np.ndarray:
    classes = np.arange(len(unique_classes))
    y_classes = get_class_from_success_rate(
        success_rates=success_rates, unique_classes=unique_classes
    )
    return compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_classes,
    )
def get_unique_classes(success_rates: np.ndarray) -> np.ndarray:
    # [0.5, 0.2, 0.1, 0.5, 0.0, 0.0, ...] => [0, 0.1, 0.2, ...]
    assert success_rates.ndim == 1
    unique_classes = np.unique(success_rates)
    return np.sort(unique_classes)
unique_labels = get_unique_classes(passed_sim)
# argmax required to make binary classes
class_weights_np = _compute_class_weight(
    success_rates=passed_sim, unique_classes=unique_labels
)
# loss_fn = WeightedSoftmaxL1(
#     unique_label_weights=torch.tensor(class_weights_np).float().to(device),
#     unique_labels=torch.tensor(unique_labels).float().to(device),
#     reduction="none",
# )



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

plt.scatter(trans[:, 0], trans[:, 1], s=1, c=passed_sim)
plt.title("Passed Sim")

# %%
preds = []
for inputs, labels in all_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    preds.extend(outputs[:, 1].tolist())

# %%
plt.scatter(trans[:, 0], trans[:, 1], s=1, c=preds)
plt.title("Predictions")

# %%
diffs = np.abs(passed_sim - preds)

# %%
plt.scatter(trans[:, 0], trans[:, 1], s=1, c=diffs)
plt.title("Diffs")
plt.colorbar()


# %%

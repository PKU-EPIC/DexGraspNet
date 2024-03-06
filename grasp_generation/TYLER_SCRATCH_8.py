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

N_data = passed_sim.shape[0]
print(f"N_data = {N_data}")

assert passed_sim.shape == (N_data,)
assert passed_penetration.shape == (N_data,)
assert passed_eval.shape == (N_data,)
assert trans.shape == (N_data, 3)
assert rot.shape == (N_data, 3, 3)

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
plt.scatter(trans[:, 0], trans[:, 1], s=1, c=passed_eval)
plt.title("Passed Eval")

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
        # labels = torch.tensor(self.passed_sim[idx]).float().reshape(1)
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

N = N_data

train_size = int(0.8 * N)
val_size = N - train_size
all_dataset = GraspDataset(
    trans=trans,
    rot=rot,
    passed_sim=passed_sim,
    passed_penetration=passed_penetration,
    passed_eval=passed_eval,
)
train_dataset, val_dataset = torch.utils.data.random_split(all_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
all_loader = DataLoader(all_dataset, batch_size=64, shuffle=False)

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
plt.scatter(trans[:, 0], trans[:, 1], s=1, c=passed_eval)
plt.colorbar()
plt.title("Passed Eval")

# %%
preds = []
for inputs, labels in all_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    preds.extend(outputs[:, 1].tolist())


# %%
plt.scatter(trans[:, 0], trans[:, 1], s=1, c=preds)
plt.colorbar()
plt.title("Predictions")


# %%
diffs = np.abs(passed_eval - preds)

# %%
plt.scatter(trans[:, 0], trans[:, 1], s=1, c=diffs)
plt.title("Diffs")
plt.colorbar()

# %%
grad_norms = []
for inputs, labels in all_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    inputs.requires_grad_(True)
    outputs = model(inputs)
    outputs.sum().backward()
    grad = inputs.grad.clone()
    grad_norm = grad.norm(dim=1)
    grad_norms.extend(grad_norm.tolist())

# %%
plt.scatter(trans[:, 0], trans[:, 1], s=1, c=np.power(grad_norms, 1/4))
plt.colorbar()
plt.title("Grad norms")

# %%
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(
    go.Scatter3d(
        x=trans[:, 0],
        y=trans[:, 1],
        z=grad_norms,
        mode="markers",
        marker=dict(size=1, color=grad_norms),
    )
)
# Set title
fig.update_layout(title_text="Grad Norms")
fig.show()

# %%
print([i for i in range(len(preds)) if 0.4 > preds[i] > 0.2])

# %%
model.eval()
start_idx = 7124
start_point = np.concatenate([trans[start_idx], rot[start_idx].flatten()])
start_point_torch = torch.tensor(start_point).float().to(device)
start_point_torch = start_point_torch.requires_grad_(True)

plt.scatter(trans[:, 0], trans[:, 1], s=1, c=preds)
plt.scatter(start_point[0], start_point[1], s=100, c="red")
pred = model(start_point_torch)[1]
pred.backward()
grad = start_point_torch.grad.clone()
grad_normalized = grad / grad.norm() * 0.01
plt.quiver(start_point[0], start_point[1], grad_normalized[0].item(), grad_normalized[1].item(), scale=0.1)

plt.title(f"Predictions, pred = {pred.item()}, actual = {passed_sim[start_idx]}")

# %%
point_torch = start_point_torch.detach().clone()
# best_point_optimizer = Adam([point_torch], lr=0.01)
pred_list = []
point_list = []
grad_list = []
for i in range(100):
    point_torch.requires_grad_(True)
    point_torch.grad = None
    pred = model(point_torch)[1]
    pred.backward()
    pred_list.append(pred.item())
    point_list.append(point_torch.tolist()[:2])
    grad_list.append(point_torch.grad.tolist()[:2])
    with torch.no_grad():
        point_torch = point_torch + point_torch.grad * 0.0001

point_list.append(point_torch.tolist()[:2])

# %%
point_arr = np.array(point_list)
grad_arr = np.array(grad_list)
print(f"point_arr.shape = {point_arr.shape}")
print(f"grad_arr.shape = {grad_arr.shape}")
# (passed_sim == 1).nonzero()

# %%
plt.scatter(trans[:, 0], trans[:, 1], s=1, c=preds)
plt.scatter(start_point[0], start_point[1], s=100, c="red")
end_point = point_list[-1]
plt.scatter(end_point[0], end_point[1], s=100, c="green")
for i in range(len(point_list) - 1):
    grad_scaled = grad_arr[i] * 0.0001
    plt.scatter(point_arr[i, 0], point_arr[i, 1], s=10, c="red")
    plt.quiver(point_arr[i, 0], point_arr[i, 1], grad_scaled[0], grad_scaled[1], scale=0.3)
plt.title(f"Predictions, pred = {pred.item()}, actual = {passed_sim[start_idx]}")

# %%
plt.plot(np.linalg.norm(grad_arr[:, 0:2], axis=1))
plt.title("grad norm")

# %%
plt.plot(pred_list)
plt.title("pred")

# %%

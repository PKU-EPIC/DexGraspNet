# %%
import numpy as np
import pathlib

data_path = pathlib.Path("../data/2024-03-05_softballs_idx11_augmented_pose_HALTON_no-rot/evaled_grasp_config_dicts/ddg-ycb_054_softball_0_0350.npy")
# data_path = pathlib.Path("../data/2024-03-05_softballs_idx11_augmented_pose_HALTON/evaled_grasp_config_dicts/ddg-ycb_054_softball_0_0350.npy")
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

# # HACK: Remove many of the 0 labels
# zero_indices = np.where(passed_eval == 0)[0]
# num_to_keep = int(0.2 * len(zero_indices))
# indices_to_keep = np.random.choice(zero_indices, size=num_to_keep, replace=False)
# indices_to_remove = np.setdiff1d(zero_indices, indices_to_keep)
# 
# passed_sim = np.delete(passed_sim, indices_to_remove)
# passed_penetration = np.delete(passed_penetration, indices_to_remove)
# passed_eval = np.delete(passed_eval, indices_to_remove)
# trans = np.delete(trans, indices_to_remove, axis=0)
# rot = np.delete(rot, indices_to_remove, axis=0)

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
from localscope import localscope
from typing import Tuple
@localscope.mfc
def plot_labels(trans: np.ndarray, labels: np.ndarray, title: str) -> Tuple[plt.Figure, np.ndarray]:
    N = trans.shape[0]
    assert trans.shape == (N, 3)
    assert labels.shape == (N,)
    z_min, z_max = np.min(trans[:, 2]), np.max(trans[:, 2])
    n_plots = 10
    z_list = np.linspace(z_min, z_max, n_plots + 1)
    nrows = int(np.ceil(np.sqrt(n_plots)))
    ncols = int(np.ceil(n_plots / nrows))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    axes = axes.flatten()
    for i, (z_low, z_high) in enumerate(zip(z_list[:-1], z_list[1:])):
        points_to_plot = np.logical_and(
            trans[:, 2] > z_low,
            trans[:, 2] < z_high,
        )
        axes[i].scatter(trans[points_to_plot, 0], trans[points_to_plot, 1], s=1, c=labels[points_to_plot])
        axes[i].set_title(f"z in [{z_low:.2f}, {z_high:.2f}]")
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes

# %%
fig, axes = plot_labels(trans=trans, labels=passed_eval, title="passed_eval")

# %%
fig, axes = plot_labels(trans=trans, labels=passed_sim, title="passed_sim")


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
        hidden_layers = [256] * 3
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
train_trans, train_labels = [], []
for i in range(len(train_dataset)):
    train_input, train_label = train_dataset[i]
    train_trans.append(train_input[:3])
    train_labels.append(train_label)
train_trans = torch.stack(train_trans, dim=0).numpy()
train_labels = torch.stack(train_labels, dim=0).numpy()

# %%
fig, axes = plot_labels(trans=train_trans, labels=train_labels[:, 1], title="Train")

# %%
val_trans, val_labels = [], []
for i in range(len(val_dataset)):
    val_input, val_label = val_dataset[i]
    val_trans.append(val_input[:3])
    val_labels.append(val_label)

val_trans = torch.stack(val_trans, dim=0).numpy()
val_labels = torch.stack(val_labels, dim=0).numpy()

# %%
fig, axes = plot_labels(trans=val_trans, labels=val_labels[:, 1], title="Val")

# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BasicMLP().to(device)
optimizer = Adam(model.parameters())
# loss_fn = nn.L1Loss(reduction="none")
loss_fn = nn.MSELoss(reduction="none")


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
fig, axes = plot_labels(trans=trans, labels=passed_eval, title="passed_eval")

# %%
preds = []
for inputs, labels in all_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    preds.extend(outputs[:, 1].tolist())


# %%
fig, axes = plot_labels(trans=trans, labels=np.array(preds), title="preds")


# %%
diffs = np.abs(passed_eval - preds)
fig, axes = plot_labels(trans=trans, labels=np.array(diffs), title="diffs")

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
fig, axes = plot_labels(trans=trans, labels=np.array(grad_norms), title="grad_norms")

# %%
print([i for i in range(len(preds)) if 1.0 > preds[i] > 0.5])

# %%
model.eval()
# start_idx = 9000
# start_idx = 9402
# start_idx = 1213
# start_idx = 1234
start_idx = 535
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
from tqdm import tqdm
consecutive_high = 0
for i in tqdm(range(10000)):
    point_torch.requires_grad_(True)
    point_torch.grad = None
    pred = model(point_torch)[1]
    pred.backward()
    pred_list.append(pred.item())
    point_list.append(point_torch.tolist()[:3])
    grad_list.append(point_torch.grad.tolist()[:3])
    with torch.no_grad():
        step_size = 0.0001
        # step_size = 0.1
        # if point_torch.grad[:3].norm() <= 1e-1:
        #     print("SMALL")
        #     print(f"point_torch.grad[:3].norm() = {point_torch.grad[:3].norm()}")
        #     step_size = 0.1 / point_torch.grad[:3].norm()
        # else:
        #     print(f"point_torch.grad[:3].norm() = {point_torch.grad[:3].norm()}")
        # print(f"point_torch.grad[:3].norm() = {point_torch.grad[:3].norm()}")
        update_step = point_torch.grad[:3] * step_size
        max_magnitude = 0.001
        update_step_magnitude = update_step.norm()
        if update_step_magnitude > max_magnitude:
            update_step = update_step / update_step_magnitude * max_magnitude
        point_torch[:3] = point_torch[:3] + update_step
    if pred.item() > 0.9:
        consecutive_high += 1
    else:
        consecutive_high = 0
    if consecutive_high > 100:
        break
    if i % 1000 == 0:
        print(f"i = {i}, pred = {pred.item()}")
        # point_torch = point_torch + point_torch.grad * 0.001

point_list.append(point_torch.tolist()[:3])

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
for i in tqdm(range(len(point_list) - 1)):
    if i % 100 != 0:
        continue
    grad_scaled = grad_arr[i] * 0.0001
    plt.scatter(point_arr[i, 0], point_arr[i, 1], s=10, c="red")
    plt.quiver(point_arr[i, 0], point_arr[i, 1], grad_scaled[0], grad_scaled[1], scale=1.0)
plt.title(f"Predictions, pred = {pred_list[-1]}")

# %%
def plot_labels_and_points(trans: np.ndarray, labels: np.ndarray, title: str, points: np.ndarray) -> Tuple[plt.Figure, np.ndarray]:
    N2 = points.shape[0]
    alphas = np.linspace(0.0, 1.0, N2)
    assert points.shape == (N2, 3)
    cmap = plt.get_cmap('rainbow')  # Choosing a colormap

    N = trans.shape[0]
    assert trans.shape == (N, 3)
    assert labels.shape == (N,)
    z_min, z_max = np.min(trans[:, 2]), np.max(trans[:, 2])
    n_plots = 10
    z_list = np.linspace(z_min, z_max, n_plots + 1)
    nrows = int(np.ceil(np.sqrt(n_plots)))
    ncols = int(np.ceil(n_plots / nrows))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    axes = axes.flatten()
    for i, (z_low, z_high) in enumerate(zip(z_list[:-1], z_list[1:])):
        points_to_plot = np.logical_and(
            trans[:, 2] > z_low,
            trans[:, 2] < z_high,
        )
        axes[i].scatter(trans[points_to_plot, 0], trans[points_to_plot, 1], s=1, c=labels[points_to_plot])
        axes[i].set_title(f"z in [{z_low:.2f}, {z_high:.2f}]")
        points2_to_plot = np.logical_and(
            points[:, 2] > z_low,
            points[:, 2] < z_high,
        )
        if points2_to_plot.sum() > 0:
            colors = cmap(alphas[points2_to_plot])
            axes[i].scatter(points[points2_to_plot, 0], points[points2_to_plot, 1], s=10, c=colors)
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


fig, axes = plot_labels_and_points(trans=trans, labels=np.array(preds), title="preds", points=point_arr)

# %%
plt.plot(np.linalg.norm(grad_arr[:, 0:3], axis=1))
plt.title("grad norm")

# %%
plt.plot(np.linalg.norm(grad_arr[:, 0:3], axis=1))
plt.title("grad norm")

# %%
plt.plot(pred_list)
plt.title("pred")

# %%

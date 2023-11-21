# %%
import numpy as np
import pathlib

# %%
object_code_and_scale_str = "ddg-gd_rubik_cube_poisson_004_0_1000"
cpu_1_grasp_path = pathlib.Path(
    "../data/2023-11-21_rubikscube_one_object/evaled_grasp_config_dicts_rerun_nonoise_cpu"
)
gpu_1_grasp_path = pathlib.Path(
    "../data/2023-11-21_rubikscube_one_object/evaled_grasp_config_dicts_rerun_nonoise_gpu"
)
cpu_10_grasps_path = pathlib.Path(
    "../data/2023-11-21_rubikscube_one_object/evaled_grasp_config_dicts_rerun_multinoise_4_cpu"
)
gpu_10_grasps_path = pathlib.Path(
    "../data/2023-11-21_rubikscube_one_object/evaled_grasp_config_dicts_rerun_multinoise_4"
)
gpu_4_grasps_path = pathlib.Path(
    "../data/2023-11-21_rubikscube_one_object/evaled_grasp_config_dicts_rerun_multinoise_4grasps_gpu"
)

assert cpu_1_grasp_path.exists()
assert gpu_1_grasp_path.exists()
assert cpu_10_grasps_path.exists()
assert gpu_10_grasps_path.exists()
assert gpu_4_grasps_path.exists()

# %%
cpu_1_grasp_filepath = cpu_1_grasp_path / f"{object_code_and_scale_str}.npy"
gpu_1_grasp_filepath = gpu_1_grasp_path / f"{object_code_and_scale_str}.npy"
cpu_10_grasps_filepath = cpu_10_grasps_path / f"{object_code_and_scale_str}.npy"
gpu_10_grasps_filepath = gpu_10_grasps_path / f"{object_code_and_scale_str}.npy"
gpu_4_grasps_filepath = gpu_4_grasps_path / f"{object_code_and_scale_str}.npy"

assert cpu_1_grasp_filepath.exists()
assert gpu_1_grasp_filepath.exists()
assert cpu_10_grasps_filepath.exists()
assert gpu_10_grasps_filepath.exists()
assert gpu_4_grasps_filepath.exists()

# %%
cpu_1_grasp_data = np.load(cpu_1_grasp_filepath, allow_pickle=True).item()
gpu_1_grasp_data = np.load(gpu_1_grasp_filepath, allow_pickle=True).item()
cpu_10_grasps_data = np.load(cpu_10_grasps_filepath, allow_pickle=True).item()
gpu_10_grasps_data = np.load(gpu_10_grasps_filepath, allow_pickle=True).item()
gpu_4_grasps_data = np.load(gpu_4_grasps_filepath, allow_pickle=True).item()

# %%
cpu_1_grasp_passed_sims = cpu_1_grasp_data["passed_simulation"]
gpu_1_grasp_passed_sims = gpu_1_grasp_data["passed_simulation"]
cpu_10_grasps_passed_sims = cpu_10_grasps_data["passed_simulation"]
gpu_10_grasps_passed_sims = gpu_10_grasps_data["passed_simulation"]
gpu_4_grasps_passed_sims = gpu_4_grasps_data["passed_simulation"]

assert cpu_1_grasp_passed_sims.shape == gpu_1_grasp_passed_sims.shape == (cpu_1_grasp_passed_sims.shape[0],)
assert cpu_10_grasps_passed_sims.shape == gpu_10_grasps_passed_sims.shape == (cpu_10_grasps_passed_sims.shape[0],)

# %%
PRINT_FIRST_K = 10
print(f"CPU 1 grasp: {cpu_1_grasp_passed_sims.shape}, {cpu_1_grasp_passed_sims[:PRINT_FIRST_K]}")
print(f"GPU 1 grasp: {gpu_1_grasp_passed_sims.shape}, {gpu_1_grasp_passed_sims[:PRINT_FIRST_K]}")
print(f"CPU 10 grasps: {cpu_10_grasps_passed_sims.shape}, {cpu_10_grasps_passed_sims[:PRINT_FIRST_K]}")
print(f"GPU 10 grasps: {gpu_10_grasps_passed_sims.shape}, {gpu_10_grasps_passed_sims[:PRINT_FIRST_K]}")
print(f"GPU 4 grasps: {gpu_4_grasps_passed_sims.shape}, {gpu_4_grasps_passed_sims[:PRINT_FIRST_K]}")

# %%
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Create subplots
fig = make_subplots(rows=2, cols=1)

# Add CPU histogram
fig.add_trace(go.Histogram(x=cpu_10_grasps_passed_sims, name='CPU'), row=1, col=1)

# Add GPU histogram
fig.add_trace(go.Histogram(x=gpu_10_grasps_passed_sims, name='GPU'), row=2, col=1)

# Update layout, if needed
fig.update_layout(height=600, width=600, title_text="CPU vs GPU Grasps Histograms")
fig.update_xaxes(title_text="Values", row=2, col=1)
fig.update_yaxes(title_text="Count", row=1, col=1)
fig.update_yaxes(title_text="Count", row=2, col=1)

# Show the plot
fig.show()

# %%
N_PTS = 20
# X-axis indices
indices = np.arange(len(cpu_10_grasps_passed_sims[:N_PTS]))

# Create the figure
fig = go.Figure()

# Add CPU bars
fig.add_trace(go.Bar(
    x=indices - 0.15,  # Slight offset to the left
    y=cpu_10_grasps_passed_sims,
    name='CPU',
    marker_color='blue',
    width=0.3  # Width of the bars
))

# Add GPU bars
fig.add_trace(go.Bar(
    x=indices + 0.15,  # Slight offset to the right
    y=gpu_10_grasps_passed_sims,
    name='GPU',
    marker_color='red',
    width=0.3  # Width of the bars
))

# Update layout
fig.update_layout(
    title="Comparison of CPU and GPU Values",
    xaxis_title="Index",
    yaxis_title="Value",
    barmode='group',  # Group bars
    width=800,
    height=600
)

# Show the plot
fig.show()

# %%
# Adding a small amount of random noise (jitter)
noise_strength = 0.01  # Adjust this value as needed
cpu_jitter = cpu_10_grasps_passed_sims + np.random.normal(0, noise_strength, cpu_10_grasps_passed_sims.shape)
gpu_jitter = gpu_10_grasps_passed_sims + np.random.normal(0, noise_strength, gpu_10_grasps_passed_sims.shape)

# Create the scatter plot with jitter
fig = go.Figure(data=go.Scatter(
    x=gpu_jitter,
    y=cpu_jitter,
    mode='markers',
    marker=dict(size=5)  # Smaller marker size
))

# Update layout
fig.update_layout(
    title="CPU vs GPU Values Scatter Plot with Jitter",
    xaxis_title="GPU Values (with Jitter)",
    yaxis_title="CPU Values (with Jitter)",
    width=800,
    height=600
)

# Show the plot
fig.show()

# %%
from scipy import stats

cpu_data = cpu_10_grasps_passed_sims 
gpu_data = gpu_10_grasps_passed_sims
# Calculate linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(gpu_data, cpu_data)

# Calculate R^2 value
r_squared = r_value**2

# Create the scatter plot
fig = go.Figure()

# Add scatter trace for data points
fig.add_trace(go.Scatter(x=gpu_data, y=cpu_data, mode='markers', name='Data'))

# Add line trace for linear regression line
fig.add_trace(go.Scatter(x=gpu_data, y=intercept + slope * gpu_data, mode='lines', name='Fit'))

# Annotate with R^2 value
fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper",
                   text=f"R^2 = {r_squared:.2f}", showarrow=False)

# Update layout
fig.update_layout(title="CPU vs GPU Correlation with Linear Fit",
                  xaxis_title="GPU",
                  yaxis_title="CPU",
                  width=800,
                  height=600)

# Show plot
fig.show()


# %%
THRESHOLD = 0.5
if THRESHOLD is not None:
    cpu_10_grasps_passed_sims = cpu_10_grasps_passed_sims > THRESHOLD
    gpu_10_grasps_passed_sims = gpu_10_grasps_passed_sims > THRESHOLD

# %%
PRINT_FIRST_K = 10
print(f"CPU 1 grasp: {cpu_1_grasp_passed_sims.shape}, {cpu_1_grasp_passed_sims[:PRINT_FIRST_K]}")
print(f"GPU 1 grasp: {gpu_1_grasp_passed_sims.shape}, {gpu_1_grasp_passed_sims[:PRINT_FIRST_K]}")
print(f"CPU 10 grasps: {cpu_10_grasps_passed_sims.shape}, {cpu_10_grasps_passed_sims[:PRINT_FIRST_K]}")
print(f"GPU 10 grasps: {gpu_10_grasps_passed_sims.shape}, {gpu_10_grasps_passed_sims[:PRINT_FIRST_K]}")

# %%
def analyze_boolean_arrays(arr1: np.ndarray, arr2: np.ndarray) -> None:
    if arr1.shape != arr2.shape:
        print("Arrays have different shapes. Analysis might not be meaningful.")
        return

    # Count where both are true
    both_true = np.sum(np.logical_and(arr1, arr2))

    # Count where arr1 is True and arr2 is False
    true_in_arr1_only = np.sum(np.logical_and(arr1, np.logical_not(arr2)))

    # Count where arr2 is True and arr1 is False
    true_in_arr2_only = np.sum(np.logical_and(arr2, np.logical_not(arr1)))

    # Count where both are false
    both_false = np.sum(np.logical_not(np.logical_or(arr1, arr2)))

    # Total count
    total = arr1.size

    print(f"Total elements: {total}")
    print(f"Both true: {both_true} ({both_true/total*100:.2f}%)")
    print(f"True in arr1 only: {true_in_arr1_only} ({true_in_arr1_only/total*100:.2f}%)")
    print(f"True in arr2 only: {true_in_arr2_only} ({true_in_arr2_only/total*100:.2f}%)")
    print(f"Both false: {both_false} ({both_false/total*100:.2f}%)")

# %%
analyze_boolean_arrays(cpu_1_grasp_passed_sims, gpu_1_grasp_passed_sims)

# %%
analyze_boolean_arrays(cpu_10_grasps_passed_sims, gpu_10_grasps_passed_sims)

# %%
analyze_boolean_arrays(cpu_1_grasp_passed_sims, cpu_10_grasps_passed_sims)

# %%
analyze_boolean_arrays(gpu_1_grasp_passed_sims, gpu_10_grasps_passed_sims)

# %%

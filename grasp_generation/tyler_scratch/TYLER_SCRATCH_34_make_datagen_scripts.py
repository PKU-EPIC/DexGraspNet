# %%
import subprocess
import pathlib

# %%
script1 = pathlib.Path("../cluster_scripts/2024-04-13_datagen_0_big.sh")
script2 = pathlib.Path("../cluster_scripts/2024-04-13_datagen_0_bigger.sh")
assert script1.exists()
assert script2.exists()

# %%
NUM_COPIES = 10
for i in range(1, NUM_COPIES + 1):
    output_script1 = script1.parent / (script1.name.replace("_0", f"_{str(i)}"))
    output_script2 = script2.parent / (script2.name.replace("_0", f"_{str(i)}"))
    cp_command1 = f"cp {script1} {output_script1}"
    cp_command2 = f"cp {script2} {output_script2}"
    subprocess.run(cp_command1, shell=True, check=True)
    subprocess.run(cp_command2, shell=True, check=True)

    sed_command1 = f"sed -i 's/_0/_{i}/g' {output_script1}"
    sed_command2 = f"sed -i 's/_0/_{i}/g' {output_script2}"
    subprocess.run(sed_command1, shell=True, check=True)
    subprocess.run(sed_command2, shell=True, check=True)

# %%

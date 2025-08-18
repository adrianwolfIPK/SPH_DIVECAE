import os
import shutil
import zipfile
import sys
from tqdm import tqdm

"""
Put the downloaded zip-files in the same folder as this script.
Adjust the root directory to the parent folder of the script/zip folders to extract.
Should be able to extract multiple fluids as well, did not test yet.
"""

# -------------------- #
# -- Configure this -- #
# -------------------- #
root_dir = str(input("Path of directory with Zip-Folder: ")) # root directory

dest_dir_name = "merged" # name of directory where files will be extracted, can be chosen freely

# ------------------------ #
# -- don't change below -- #
# ------------------------ #
fluids_target = os.path.join(root_dir, dest_dir_name, "fluids")
walls_target = os.path.join(root_dir, dest_dir_name, "walls")
sensors_target = os.path.join(root_dir, dest_dir_name, "sensors")
rec_surf_target = os.path.join(root_dir, dest_dir_name, "reconstucted_surfaces")

destination = os.path.join(root_dir, dest_dir_name)

os.makedirs(fluids_target, exist_ok=True)
os.makedirs(walls_target, exist_ok=True)
os.makedirs(sensors_target, exist_ok=True)
os.makedirs(rec_surf_target, exist_ok=True)

def copy_fluids(root_dir=root_dir):
    print("\n -- Copying Fluid Data -- \n")

    for dir in os.listdir(root_dir):
        path = os.path.join(root_dir, dir)
        if not os.path.isdir(path):
            continue

        fluids_path = os.path.join(path, "Fluids")
        if not os.path.isdir(fluids_path):
            print(f"No Fluids folder in {path}")
            continue

        for fluid_dir in os.listdir(fluids_path):
            subfolder_path = os.path.join(fluids_path, fluid_dir)
            if not os.path.isdir(subfolder_path):
                continue

            for f in os.listdir(subfolder_path):
                if not f.endswith("_0.vtp"):
                    continue

                full_path = os.path.join(subfolder_path, f)
                base = f[:-6]  # remove "_0.vtp"

                parts = base.split("_")
                if len(parts) < 2:
                    print(f"Unexpected Filename Format: {f}")
                    continue

                timestep_str = parts[-1]

                try:
                    timestep = int(timestep_str)
                except ValueError:
                    print(f"Invalid Timestep in Filename: {f}")
                    continue

                dest_normal = os.path.join(fluids_target, f"{base}.vtp")
                shutil.copy(full_path, dest_normal)
                # print(f"Copied: {full_path} -> {dest_normal}")

                if timestep == 1:
                    parts[-1] = "0"
                    timestep0_name = "_".join(parts) + ".vtp"
                    dest_timestep0 = os.path.join(fluids_target, timestep0_name)
                    shutil.copy(full_path, dest_timestep0)
                    print(f"Copied Timestep 1 as Timestep 0 and 1: {full_path} -> {dest_timestep0}")

def copy_walls(root_dir=root_dir):
    print("\n -- Copying Wall Data -- \n")
    for dir in os.listdir(root_dir):
        path = os.path.join(root_dir, dir)
        if not os.path.isdir(path):
            continue

        walls_path = os.path.join(path, "Walls")

        if not os.path.isdir(walls_path):
            continue

        for f in os.listdir(walls_path):
            if f.endswith(".vtu"):
                src_file = os.path.join(walls_path, f)
                dest_file = os.path.join(walls_target, f)

                shutil.move(src_file, dest_file)
                # print(f"Copied: {src_file} -> {dest_file}")

def copy_sensors(root_dir=root_dir):
    print("\n -- Copying Sensor Data -- \n")
    for block_dir in os.listdir(root_dir):
        path = os.path.join(root_dir, block_dir)
        if not os.path.isdir(path):
            continue

        sensors_path = os.path.join(path, "Sensors")

        if not os.path.isdir(sensors_path):
            continue

        for fname in sorted(os.listdir(sensors_path)):
            source = os.path.join(sensors_path, fname)
            if os.path.isfile(source):
                destination = os.path.join(sensors_target, fname)

                shutil.move(source, destination)
                # print(f"Copied: {source} -> {destination}")

def copy_rec_surf(root_dir=root_dir):
    print("\n -- Copying Reconstructed Surface Data -- \n")
    for dir in os.listdir(root_dir):
        path = os.path.join(root_dir, dir)
        if not os.path.isdir(path):
            continue

        rec_surf_path = os.path.join(path, "Reconstructed_Surfaces")

        if not os.path.isdir(rec_surf_path):
            continue

        for f in os.listdir(rec_surf_path):
            if f.endswith(".vtu"):
                src_file = os.path.join(rec_surf_path, f)
                dest_file = os.path.join(rec_surf_target, f)

                shutil.move(src_file, dest_file)
                # print(f"Copied: {src_file} -> {dest_file}")

def unzip_dirs(root_dir=root_dir):
    zips = sorted([item for item in os.listdir(root_dir) if item.lower().endswith(".zip")])
    total = len(zips)
    if total == 0:
        print("No ZIPs found.")
        return

    print("\n-- Starting to unzip data --\n")
    for item in tqdm(zips, desc="Unzipping", unit="zip", ncols=60):
        zip_path = os.path.join(root_dir, item)
        out_folder = os.path.join(root_dir, os.path.splitext(item)[0])
        os.makedirs(out_folder, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(out_folder)
        except zipfile.BadZipFile:
            tqdm.write(f"Error: {item} is not a valid zip file.")


def unzip_with_progress(root_dir=root_dir):
    zips = [item for item in os.listdir(root_dir) if item.lower().endswith(".zip")]
    total = len(zips)
    if total == 0:
        print("No ZIPs found.")
        return

    bar_width = 40

    print("\n-- Starting to unzip Data --")
    for idx, item in enumerate(zips, start=1):
        zip_path = os.path.join(root_dir, item)
        out_folder = os.path.join(root_dir, os.path.splitext(item)[0])
        os.makedirs(out_folder, exist_ok=True)

        # print(f"\nUnzipping {item} into {out_folder}...")

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(out_folder)
            print(f"Done: {item}")
        except zipfile.BadZipFile:
            print(f"Error: {item} is not a valid zip file.")
        finally:
            # update progress bar
            done = idx / total
            filled = int(bar_width * done)
            bar = "=" * filled + "-" * (bar_width - filled)
            percent_str = f"{done * 100:5.1f}%"
            sys.stdout.write(f"\rProgress: [{bar}] {percent_str}")
            sys.stdout.flush()

    print("\n -- All zips extracted. -- \n")

if __name__ == "__main__":
    # unzip_with_progress() # use when tqdm is not installed
    
    unzip_dirs() # using tqdm lib, install with 'pip install tqdm'
    copy_fluids()
    copy_walls()
    copy_sensors()
    copy_rec_surf()
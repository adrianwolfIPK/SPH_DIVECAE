# -*- coding: utf-8 -*-

APP_VERSION = "1.0.1"

import os
import shutil
import zipfile
import sys

from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk

# -------------------- #
# ------- GUI -------- #
# -------------------- #
ctk.set_appearance_mode("System")  # "Dark", "Light", or "System"
ctk.set_default_color_theme("blue")

class MergeApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Data Merger")
        self.geometry("800x600")
        self.root_dir = ""

        # title
        ctk.CTkLabel(self, text="Select Folder and Data to Merge", font=("Arial", 18)).pack(pady=15)

        # folder selection
        ctk.CTkButton(self, text="Select Folder", command=self.select_folder).pack(pady=10)

        self.path_label = ctk.CTkLabel(self, text="No folder selected", text_color="blue")
        self.path_label.pack()

        # checkboxes
        self.fluid_var = ctk.BooleanVar(value=True)
        self.wall_var = ctk.BooleanVar(value=True)
        self.sensor_var = ctk.BooleanVar(value=True)
        self.recsurf_var = ctk.BooleanVar(value=True)

        self.fluid_cb = ctk.CTkCheckBox(self, text="Fluids", variable=self.fluid_var)
        self.wall_cb = ctk.CTkCheckBox(self, text="Walls", variable=self.wall_var)
        self.sensor_cb = ctk.CTkCheckBox(self, text="Sensors", variable=self.sensor_var)
        self.recsurf_cb = ctk.CTkCheckBox(self, text="Reconstructed Surfaces", variable=self.recsurf_var)

        self.fluid_cb.pack(pady=3)
        self.wall_cb.pack(pady=3)
        self.sensor_cb.pack(pady=3)
        self.recsurf_cb.pack(pady=3)

        # start Button
        self.start_button = ctk.CTkButton(self, text="Start Merge", command=self.start_merge)
        self.start_button.pack(pady=20)

        # status Label
        self.status_label = ctk.CTkLabel(self, text="Status: Waiting", font=("Arial", 14))
        self.status_label.pack(pady=(10, 2))

        # progress bar
        self.progress = ctk.CTkProgressBar(self, height=20)
        self.progress.set(0)
        self.progress.pack(pady=10, fill="x", padx=50)

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select Folder with ZIPs or Extracted Folders")
        if folder:
            self.root_dir = folder
            self.path_label.configure(text=folder, text_color="blue")

    def start_merge(self):
        if not self.root_dir:
            self.status_label.configure(text="Please select a folder first.", text_color="red")
            return

        self.status_label.configure(text="Starting merge...", text_color="blue")

        # define destination folders
        dest_dir_name = "merged"
        fluids_target = os.path.join(self.root_dir, dest_dir_name, "fluids")
        walls_target = os.path.join(self.root_dir, dest_dir_name, "walls")
        sensors_target = os.path.join(self.root_dir, dest_dir_name, "sensors")
        rec_surf_target = os.path.join(self.root_dir, dest_dir_name, "reconstucted_surfaces")

        # create dirs
        os.makedirs(fluids_target, exist_ok=True)
        os.makedirs(walls_target, exist_ok=True)
        os.makedirs(sensors_target, exist_ok=True)
        os.makedirs(rec_surf_target, exist_ok=True)

        self.status_label.configure(text="Status: Unzipping files...")
        self.update_idletasks()
        unzip_dirs(self.root_dir, progress_bar=self.progress, app=self)

        if self.fluid_var.get():
            self.status_label.configure(text="Status: Copying fluids...")
            self.update_idletasks()
            copy_fluids(self.root_dir, fluids_target, self.progress, self)

        if self.wall_var.get():
            self.status_label.configure(text="Status: Copying walls...")
            self.update_idletasks()
            copy_walls(self.root_dir, walls_target, self.progress, self)

        if self.sensor_var.get():
            self.status_label.configure(text="Status: Copying sensors...")
            self.update_idletasks()
            copy_sensors(self.root_dir, sensors_target, self.progress, self)

        if self.recsurf_var.get():
            self.status_label.configure(text="Status: Copying reconstructed surfaces...")
            self.update_idletasks()
            copy_rec_surf(self.root_dir, rec_surf_target, self.progress, self)

        self.progress.set(1)
        self.status_label.configure(text="Merge completed!", text_color="green")

# # -------------------- #
# # -- Configure this -- #
# # -------------------- #
# dest_dir_name = "merged" # name of directory where files will be extracted, can be chosen freely

# ------------------------ #
# -- don't change below -- #
# ------------------------ #
# root_dir = str(input("Path of directory with Zip-Folder: ")) # root directory
# root = tk.Tk()
# root.withdraw()  # Hide the root window
# root_dir = filedialog.askdirectory(title="Select Directory with ZIP Files to unpack and merge")
# if not root_dir:
#     print("No folder selected. Exiting.")
#     sys.exit(1)

# fluids_target = os.path.join(root_dir, dest_dir_name, "fluids")
# walls_target = os.path.join(root_dir, dest_dir_name, "walls")
# sensors_target = os.path.join(root_dir, dest_dir_name, "sensors")
# rec_surf_target = os.path.join(root_dir, dest_dir_name, "reconstucted_surfaces")

# destination = os.path.join(root_dir, dest_dir_name)

# os.makedirs(fluids_target, exist_ok=True)
# os.makedirs(walls_target, exist_ok=True)
# os.makedirs(sensors_target, exist_ok=True)
# os.makedirs(rec_surf_target, exist_ok=True)

def copy_fluids(root_dir, target, progress_bar=None, app=None):
    fluid_folders = []
    for dir in os.listdir(root_dir):
        fluids_path = os.path.join(root_dir, dir, "Fluids")
        if os.path.isdir(fluids_path):
            fluid_folders.append(fluids_path)

    total = len(fluid_folders)
    if total == 0:
        return

    for i, fluids_path in enumerate(tqdm(fluid_folders, desc="Copying Fluids", ncols=60)):
        for fluid_dir in os.listdir(fluids_path):
            subfolder_path = os.path.join(fluids_path, fluid_dir)
            if not os.path.isdir(subfolder_path):
                continue

            for f in os.listdir(subfolder_path):
                if not f.endswith("_0.vtp"):
                    continue

                full_path = os.path.join(subfolder_path, f)
                base = f[:-6]
                parts = base.split("_")

                if len(parts) < 2:
                    continue

                timestep_str = parts[-1]
                try:
                    timestep = int(timestep_str)
                except ValueError:
                    continue

                dest_normal = os.path.join(target, f"{base}.vtp")
                shutil.copy(full_path, dest_normal)

                if timestep == 1:
                    parts[-1] = "0"
                    timestep0_name = "_".join(parts) + ".vtp"
                    dest_timestep0 = os.path.join(target, timestep0_name)
                    shutil.copy(full_path, dest_timestep0)

        # GUI progress bar
        if progress_bar and app:
            progress_bar.set((i + 1) / total)
            app.update_idletasks()

def copy_walls(root_dir, target, progress_bar=None, app=None):
    wall_folders = []
    for dir in os.listdir(root_dir):
        walls_path = os.path.join(root_dir, dir, "Walls")
        if os.path.isdir(walls_path):
            wall_folders.append(walls_path)

    total = len(wall_folders)
    if total == 0:
        return

    for i, walls_path in enumerate(tqdm(wall_folders, desc="Copying Walls", ncols=60)):
        for f in os.listdir(walls_path):
            if f.endswith(".vtu"):
                src = os.path.join(walls_path, f)
                dst = os.path.join(target, f)
                shutil.move(src, dst)

        if progress_bar and app:
            progress_bar.set((i + 1) / total)
            app.update_idletasks()

def copy_sensors(root_dir, target, progress_bar=None, app=None):
    sensor_folders = []
    for dir in os.listdir(root_dir):
        sensors_path = os.path.join(root_dir, dir, "Sensors")
        if os.path.isdir(sensors_path):
            sensor_folders.append(sensors_path)

    total = len(sensor_folders)
    if total == 0:
        return

    for i, sensors_path in enumerate(tqdm(sensor_folders, desc="Copying Sensors", ncols=60)):
        for f in os.listdir(sensors_path):
            src = os.path.join(sensors_path, f)
            dst = os.path.join(target, f)
            if os.path.isfile(src):
                shutil.move(src, dst)

        if progress_bar and app:
            progress_bar.set((i + 1) / total)
            app.update_idletasks()

def copy_rec_surf(root_dir, target, progress_bar=None, app=None):
    recsurf_folders = []
    for dir in os.listdir(root_dir):
        rec_surf_path = os.path.join(root_dir, dir, "Reconstructed_Surfaces")
        if os.path.isdir(rec_surf_path):
            recsurf_folders.append(rec_surf_path)

    total = len(recsurf_folders)
    if total == 0:
        return

    for i, rec_surf_path in enumerate(tqdm(recsurf_folders, desc="Copying Reconstructed Surfaces", ncols=60)):
        for f in os.listdir(rec_surf_path):
            if f.endswith(".vtu"):
                src = os.path.join(rec_surf_path, f)
                dst = os.path.join(target, f)
                shutil.move(src, dst)

        if progress_bar and app:
            progress_bar.set((i + 1) / total)
            app.update_idletasks()

def unzip_dirs(root_dir, progress_bar=None, app=None):
    zips = sorted([item for item in os.listdir(root_dir) if item.lower().endswith(".zip")])
    total = len(zips)
    if not zips:
        print("No ZIP files found.")
        return

    print("\n-- Unzipping Data --\n")

    for i, item in enumerate(tqdm(zips, desc="Unzipping", unit="zip", ncols=60)):
        zip_path = os.path.join(root_dir, item)
        out_folder = os.path.join(root_dir, os.path.splitext(item)[0])
        os.makedirs(out_folder, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(out_folder)
        except zipfile.BadZipFile:
            tqdm.write(f"Invalid ZIP: {item}")
            continue

        # GUI progress bar
        if progress_bar and app:
            progress_bar.set((i + 1) / total)
            app.update_idletasks()
        
    print("\n -- All ZIPs extracted. -- \n")


# def unzip_with_progress(root_dir=root_dir): # deprecated version of unzip_dirs()
#     zips = [item for item in os.listdir(root_dir) if item.lower().endswith(".zip")]
#     total = len(zips)
#     if total == 0:
#         print("No ZIPs found.")
#         return

#     bar_width = 100

#     print("\n-- Starting to unzip Data --")
#     for idx, item in enumerate(zips, start=1):
#         zip_path = os.path.join(root_dir, item)
#         out_folder = os.path.join(root_dir, os.path.splitext(item)[0])
#         os.makedirs(out_folder, exist_ok=True)

#         # print(f"\nUnzipping {item} into {out_folder}...")

#         try:
#             with zipfile.ZipFile(zip_path, "r") as zf:
#                 zf.extractall(out_folder)
#             print(f"Done: {item}")
#         except zipfile.BadZipFile:
#             print(f"Error: {item} is not a valid zip file.")
#         finally:
#             # update progress bar
#             done = idx / total
#             filled = int(bar_width * done)
#             bar = "=" * filled + "-" * (bar_width - filled)
#             percent_str = f"{done * 100:5.1f}%"
#             sys.stdout.write(f"\rProgress: [{bar}] {percent_str}")
#             sys.stdout.flush()

#     print("\n -- All zips extracted. -- \n")

# -------------------- #
# --- Update Logic --- #
# -------------------- #

import requests

GITHUB_VERSION_URL = "https://github.com/adrianwolfIPK/SPH_DIVECAE/blob/main/post/sort_version.txt"
EXE_DOWNLOAD_URL = "https://github.com/adrianwolfIPK/SPH_DIVECAE/releases/download/v1.0.0/sort.exe"

def check_for_update(current_version):
    try:
        response = requests.get(GITHUB_VERSION_URL, timeout=5)
        latest_version = response.text.strip()

        if latest_version > current_version:
            return latest_version
    except Exception as e:
        print("Update check failed:", e)

    return None

def download_new_version(download_url, save_path):
    try:
        print("Downloading update...")
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except Exception as e:
        print("Download failed:", e)
        return False

def update_app_if_needed():
    new_version = check_for_update(APP_VERSION)
    if new_version:
        print(f"New version available: {new_version}")
        
        exe_path = sys.executable
        temp_path = exe_path + ".new"

        success = download_new_version(EXE_DOWNLOAD_URL, temp_path)
        if success:
            print("Update downloaded. Replacing application...")
            try:
                os.replace(temp_path, exe_path)
                print("Update installed. Restarting app...")
                os.execv(exe_path, sys.argv)
            except Exception as e:
                print("Failed to apply update:", e)
        else:
            print("Update download failed.")
    else:
        print("You're up to date.")

if __name__ == "__main__":
    update_app_if_needed()

    app = MergeApp()
    app.mainloop()
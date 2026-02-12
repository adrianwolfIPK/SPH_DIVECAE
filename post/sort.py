# -*- coding: utf-8 -*-

import os
import shutil
import zipfile
import sys

from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog, messagebox
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
        targets = {
            "fluids": os.path.join(self.root_dir, dest_dir_name, "fluids"),
            "walls": os.path.join(self.root_dir, dest_dir_name, "walls"),
            "sensors": os.path.join(self.root_dir, dest_dir_name, "sensors"),
            "rec_surf": os.path.join(self.root_dir, dest_dir_name, "reconstucted_surfaces"),
        }

        # create dirs
        for t in targets.values():
            os.makedirs(t, exist_ok=True)

        # get checkbox selections
        selections = {
            "fluids": self.fluid_var.get(),
            "walls": self.wall_var.get(),
            "sensors": self.sensor_var.get(),
            "rec_surf": self.recsurf_var.get(),
        }

        self.status_label.configure(text="Status: Processing ZIPs...")
        self.update_idletasks()
        process_and_merge(self.root_dir, targets, selections,
                          progress_bar=self.progress, app=self)

        self.progress.set(1)
        self.status_label.configure(text="Merge completed!", text_color="green")

def move_fluids_from_folder(folder_path, target):
    """Move fluid files from a single extracted folder to the merged target."""
    fluids_path = os.path.join(folder_path, "Fluids")
    if not os.path.isdir(fluids_path):
        return

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
            shutil.move(full_path, dest_normal)

            if timestep == 1:
                parts[-1] = "0"
                timestep0_name = "_".join(parts) + ".vtp"
                dest_timestep0 = os.path.join(target, timestep0_name)
                shutil.copy(dest_normal, dest_timestep0)

def move_walls_from_folder(folder_path, target):
    """Move wall files from a single extracted folder to the merged target."""
    walls_path = os.path.join(folder_path, "Walls")
    if not os.path.isdir(walls_path):
        return

    for f in os.listdir(walls_path):
        if f.endswith(".vtu"):
            src = os.path.join(walls_path, f)
            dst = os.path.join(target, f)
            shutil.move(src, dst)

def move_sensors_from_folder(folder_path, target):
    """Move sensor files from a single extracted folder to the merged target."""
    sensors_path = os.path.join(folder_path, "Sensors")
    if not os.path.isdir(sensors_path):
        return

    for f in os.listdir(sensors_path):
        src = os.path.join(sensors_path, f)
        dst = os.path.join(target, f)
        if os.path.isfile(src):
            shutil.move(src, dst)

def move_rec_surf_from_folder(folder_path, target):
    """Move reconstructed surface files from a single extracted folder to the merged target."""
    rec_surf_path = os.path.join(folder_path, "Reconstructed_Surfaces")
    if not os.path.isdir(rec_surf_path):
        return

    for f in os.listdir(rec_surf_path):
        if f.endswith(".vtu"):
            src = os.path.join(rec_surf_path, f)
            dst = os.path.join(target, f)
            shutil.move(src, dst)

def move_data_from_folder(folder, targets, selections):
    """Move selected data from an extracted folder into merged targets."""
    if selections.get("fluids"):
        move_fluids_from_folder(folder, targets["fluids"])
    if selections.get("walls"):
        move_walls_from_folder(folder, targets["walls"])
    if selections.get("sensors"):
        move_sensors_from_folder(folder, targets["sensors"])
    if selections.get("rec_surf"):
        move_rec_surf_from_folder(folder, targets["rec_surf"])

def process_and_merge(root_dir, targets, selections, progress_bar=None, app=None):
    """Process each ZIP one at a time: extract -> move data -> delete folder -> delete ZIP."""
    dest_dir_name = "merged"

    # Resume: recover leftover folders from a previous interrupted run
    # Move any remaining data to merged first, then delete the folder
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path) and item != dest_dir_name:
            print(f"Recovering leftover folder: {item}")
            move_data_from_folder(item_path, targets, selections)
            shutil.rmtree(item_path)
            print(f"Recovered and deleted: {item}")

    zips = sorted([item for item in os.listdir(root_dir)
                   if item.lower().endswith(".zip")])
    total = len(zips)
    if not zips:
        print("No ZIP files found.")
        return

    print(f"\n-- Processing {total} ZIPs --\n")

    for i, item in enumerate(tqdm(zips, desc="Processing",
                                  unit="zip", ncols=60)):
        zip_path = os.path.join(root_dir, item)
        out_folder = os.path.join(root_dir, os.path.splitext(item)[0])
        os.makedirs(out_folder, exist_ok=True)

        # 1. Extract ZIP (keep ZIP as backup until data is safe)
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(out_folder)
        except zipfile.BadZipFile:
            tqdm.write(f"Invalid ZIP: {item}")
            continue

        # 2. Move data to merged folders
        move_data_from_folder(out_folder, targets, selections)

        # 3. Delete the extracted folder
        shutil.rmtree(out_folder)

        # 4. Delete ZIP only after data is safely in merged/
        os.remove(zip_path)

        tqdm.write(f"Done: {item}")

        if progress_bar and app:
            progress_bar.set((i + 1) / total)
            app.update_idletasks()

    print(f"\n -- All {total} ZIPs processed. -- \n")

# -------------------- #
# --- Update Logic --- #
# -------------------- #

import requests
import tempfile

APP_VERSION = "1.1.0"
REPO_OWNER = "adrianwolfIPK"
REPO_NAME = "SPH_DIVECAE"

def fetch_latest_release():
    api_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases/latest"
    resp = requests.get(api_url, timeout=5)
    resp.raise_for_status()

    data = resp.json()
    version = data["tag_name"]

    exe_url = None
    for asset in data["assets"]:
        if asset["name"].endswith(".exe"):
            exe_url = asset["browser_download_url"]
            break

    if exe_url is None:
        raise Exception("No .exe file found in the latest release.")

    return version, exe_url

def is_frozen():
    return getattr(sys, 'frozen', False)

from packaging import version

def check_for_updates_gui():
    if not is_frozen():
        print("Skipping update check (not running from .exe)")
        return

    try:
        latest_version, download_url = fetch_latest_release()
        if version.parse(latest_version) > version.parse(APP_VERSION):
            answer = messagebox.askyesno("Update Available",
                                         f"A new version ({latest_version}) is available.\nDo you want to update now?")
            if answer:
                download_and_replace(download_url)
    except Exception as e:
        print(f"Update check failed: {e}")

import subprocess
import textwrap

def download_and_replace(download_url):
    temp_dir = tempfile.gettempdir()
    new_exe_path = os.path.join(temp_dir, "new_sort.exe")
    updater_bat = os.path.join(temp_dir, "update_sort.bat")
    current_exe = sys.executable.replace('/', '\\')
    exe_name = os.path.basename(current_exe)

    try:
        # download latest exe
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(new_exe_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # create BAT script
        bat_content = textwrap.dedent(f"""\
            @echo off
            echo --- Running Updater ---
            echo Waiting for process to close: {exe_name}

            :waitloop
            tasklist | findstr /I "{exe_name}" >nul
            if not errorlevel 1 (
                timeout /t 1 >nul
                goto waitloop
            )

            echo Replacing old EXE with new one...
            move /Y "{new_exe_path}" "{current_exe}"
            echo Restarting application...
            start "" "{current_exe}"

            echo Done. Exiting...
            del "%~f0"
        """)

        with open(updater_bat, "w") as bat_file:
            bat_file.write(bat_content)

        # run BAT file in new shell and exit current app
        print(f"Launching updater BAT: {updater_bat}")
        subprocess.Popen(["cmd.exe", "/c", updater_bat], creationflags=subprocess.CREATE_NEW_CONSOLE)

        # kill current process
        sys.exit(0)

    except Exception as e:
        messagebox.showerror("Update Failed", f"Failed to install update:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw() # hide base window
    check_for_updates_gui()
    root.destroy()

    app = MergeApp()
    app.mainloop()

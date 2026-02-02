"""Test script to debug sort.py functions"""
import sys
import os

sys.path.append(r'c:\Users\adr61871\Desktop\fisherman_test\post')
from sort import unzip_dirs, copy_fluids, copy_walls, copy_sensors, copy_rec_surf, cleanup_extracted_dirs

test_dir = r"c:\Users\adr61871\Desktop\fisherman_test\static\test_data"
merged_dir = os.path.join(test_dir, "merged")

# Create merged subdirectories
fluids_target = os.path.join(merged_dir, "fluids")
walls_target = os.path.join(merged_dir, "walls")
sensors_target = os.path.join(merged_dir, "sensors")
rec_surf_target = os.path.join(merged_dir, "reconstructed_surfaces")

os.makedirs(fluids_target, exist_ok=True)
os.makedirs(walls_target, exist_ok=True)
os.makedirs(sensors_target, exist_ok=True)
os.makedirs(rec_surf_target, exist_ok=True)

print("=" * 60)
print("STEP 1: Unzipping files...")
print("=" * 60)
extracted_folders = unzip_dirs(test_dir)
print(f"\nExtracted folders: {extracted_folders}")

print("\n" + "=" * 60)
print("STEP 2: Checking what directories exist...")
print("=" * 60)
for item in os.listdir(test_dir):
    item_path = os.path.join(test_dir, item)
    if os.path.isdir(item_path):
        print(f"Directory: {item}")
        # Check subdirectories
        try:
            subdirs = [d for d in os.listdir(item_path) if os.path.isdir(os.path.join(item_path, d))]
            print(f"  Subdirectories: {subdirs}")
        except:
            pass

print("\n" + "=" * 60)
print("STEP 3: Copying fluids...")
print("=" * 60)
copy_fluids(test_dir, fluids_target)
print(f"Files in fluids_target: {len(os.listdir(fluids_target)) if os.path.exists(fluids_target) else 0}")

print("\n" + "=" * 60)
print("STEP 4: Copying walls...")
print("=" * 60)
copy_walls(test_dir, walls_target)
print(f"Files in walls_target: {len(os.listdir(walls_target)) if os.path.exists(walls_target) else 0}")

print("\n" + "=" * 60)
print("STEP 5: Copying sensors...")
print("=" * 60)
copy_sensors(test_dir, sensors_target)
print(f"Files in sensors_target: {len(os.listdir(sensors_target)) if os.path.exists(sensors_target) else 0}")

print("\n" + "=" * 60)
print("STEP 6: Copying reconstructed surfaces...")
print("=" * 60)
copy_rec_surf(test_dir, rec_surf_target)
print(f"Files in rec_surf_target: {len(os.listdir(rec_surf_target)) if os.path.exists(rec_surf_target) else 0}")

print("\n" + "=" * 60)
print("STEP 7: Checking merged folder contents...")
print("=" * 60)
for subdir in ["fluids", "walls", "sensors", "reconstructed_surfaces"]:
    subdir_path = os.path.join(merged_dir, subdir)
    if os.path.exists(subdir_path):
        files = os.listdir(subdir_path)
        print(f"\n{subdir}/: {len(files)} files")
        if files:
            print(f"  First 5: {files[:5]}")

print("\n" + "=" * 60)
print("STEP 8: Cleaning up extracted directories...")
print("=" * 60)
cleanup_extracted_dirs(extracted_folders)

print("\n" + "=" * 60)
print("DONE! Final directory structure:")
print("=" * 60)
for item in os.listdir(test_dir):
    print(f"  {item}")

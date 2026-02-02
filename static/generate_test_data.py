# -*- coding: utf-8 -*-
"""
Test data generator for sort.py
Creates test ZIP files with proper directory structure for SPH simulation data
"""

import os
import zipfile
import random

def create_dummy_vtp_file(filepath, timestep=0):
    """Create a dummy VTP (VTK PolyData) file."""
    vtp_content = f"""<?xml version="1.0"?>
<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">
  <PolyData>
    <Piece NumberOfPoints="10" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">
      <Points>
        <DataArray type="Float32" NumberOfComponents="3" format="ascii">
          {' '.join([f'{random.random():.6f}' for _ in range(30)])}
        </DataArray>
      </Points>
      <PointData>
        <DataArray type="Float32" Name="velocity" NumberOfComponents="3" format="ascii">
          {' '.join([f'{random.random():.6f}' for _ in range(30)])}
        </DataArray>
        <DataArray type="Float32" Name="pressure" format="ascii">
          {' '.join([f'{random.random():.6f}' for _ in range(10)])}
        </DataArray>
      </PointData>
    </Piece>
  </PolyData>
</VTKFile>
"""
    with open(filepath, 'w') as f:
        f.write(vtp_content)

def create_dummy_vtu_file(filepath):
    """Create a dummy VTU (VTK Unstructured Grid) file."""
    vtu_content = f"""<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="8" NumberOfCells="1">
      <Points>
        <DataArray type="Float32" NumberOfComponents="3" format="ascii">
          {' '.join([f'{random.random():.6f}' for _ in range(24)])}
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii">
          0 1 2 3 4 5 6 7
        </DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">
          8
        </DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">
          12
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""
    with open(filepath, 'w') as f:
        f.write(vtu_content)

def create_dummy_sensor_csv(filepath, timesteps=5):
    """Create a dummy sensor CSV file."""
    csv_content = "time,pressure,temperature,velocity_x,velocity_y,velocity_z\n"
    for t in range(timesteps):
        time = t * 0.001
        csv_content += f"{time:.6f},{random.random()*100:.3f},{random.random()*50+20:.2f},"
        csv_content += f"{random.random():.6f},{random.random():.6f},{random.random():.6f}\n"

    with open(filepath, 'w') as f:
        f.write(csv_content)

def create_test_simulation_data(base_dir, sim_name, num_timesteps=5):
    """
    Create a test simulation dataset with proper structure.

    Structure:
    sim_name/
    ├── Fluids/
    │   ├── fluid_1/
    │   │   ├── fluid_1_1_0.vtp
    │   │   ├── fluid_1_2_0.vtp
    │   │   └── ...
    │   └── fluid_2/
    │       └── ...
    ├── Walls/
    │   ├── wall_inlet_0.vtu
    │   └── wall_outlet_0.vtu
    ├── Sensors/
    │   ├── sensor_pressure_01.csv
    │   └── sensor_velocity_02.csv
    └── Reconstructed_Surfaces/
        ├── surface_fluid1_0.vtu
        └── surface_fluid2_0.vtu
    """
    sim_path = os.path.join(base_dir, sim_name)

    # Create directory structure
    fluids_dir = os.path.join(sim_path, "Fluids")
    walls_dir = os.path.join(sim_path, "Walls")
    sensors_dir = os.path.join(sim_path, "Sensors")
    recsurf_dir = os.path.join(sim_path, "Reconstructed_Surfaces")

    os.makedirs(fluids_dir, exist_ok=True)
    os.makedirs(walls_dir, exist_ok=True)
    os.makedirs(sensors_dir, exist_ok=True)
    os.makedirs(recsurf_dir, exist_ok=True)

    # Create Fluids data (2 fluid domains, multiple timesteps)
    for fluid_id in [1, 2]:
        fluid_subdir = os.path.join(fluids_dir, f"fluid_{fluid_id}")
        os.makedirs(fluid_subdir, exist_ok=True)

        for timestep in range(1, num_timesteps + 1):
            filename = f"fluid_{fluid_id}_{timestep}_0.vtp"
            filepath = os.path.join(fluid_subdir, filename)
            create_dummy_vtp_file(filepath, timestep)
            print(f"  Created: {filename}")

    # Create Walls data
    wall_names = ["inlet", "outlet", "boundary"]
    for wall_name in wall_names:
        filename = f"wall_{wall_name}_0.vtu"
        filepath = os.path.join(walls_dir, filename)
        create_dummy_vtu_file(filepath)
        print(f"  Created: {filename}")

    # Create Sensors data
    sensor_names = ["pressure_01", "pressure_02", "velocity_03"]
    for sensor_name in sensor_names:
        filename = f"sensor_{sensor_name}.csv"
        filepath = os.path.join(sensors_dir, filename)
        create_dummy_sensor_csv(filepath, num_timesteps)
        print(f"  Created: {filename}")

    # Create Reconstructed Surfaces
    for fluid_id in [1, 2]:
        filename = f"surface_fluid{fluid_id}_0.vtu"
        filepath = os.path.join(recsurf_dir, filename)
        create_dummy_vtu_file(filepath)
        print(f"  Created: {filename}")

    return sim_path

def create_zip_from_directory(source_dir, output_zip):
    """Create a ZIP file from a directory."""
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Use source_dir itself as base, not its parent, to avoid double nesting
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)
    print(f"\nCreated ZIP: {output_zip}")

def main():
    """Generate test data and ZIP files."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_output_dir = os.path.join(script_dir, "test_data")

    # Clean up old test data if it exists
    if os.path.exists(test_output_dir):
        import shutil
        shutil.rmtree(test_output_dir)

    os.makedirs(test_output_dir, exist_ok=True)

    print("="*60)
    print("Generating Test Data for sort.py")
    print("="*60)

    # Create 3 test simulations
    sim_names = ["simulation_001", "simulation_002", "simulation_003"]

    for sim_name in sim_names:
        print(f"\nCreating simulation: {sim_name}")
        print("-"*40)
        sim_path = create_test_simulation_data(test_output_dir, sim_name, num_timesteps=5)

        # Create ZIP file
        zip_filename = os.path.join(test_output_dir, f"{sim_name}.zip")
        create_zip_from_directory(sim_path, zip_filename)

        # Remove the unzipped directory (we only want ZIPs for testing)
        import shutil
        shutil.rmtree(sim_path)
        print(f"Removed unzipped directory: {sim_name}")

    print("\n" + "="*60)
    print("Test data generation complete!")
    print(f"Output directory: {test_output_dir}")
    print("="*60)
    print("\nYou can now use sort.py with the test_data folder:")
    print(f"  1. Run sort.py")
    print(f"  2. Select folder: {test_output_dir}")
    print(f"  3. Click 'Start Merge'")
    print("\nExpected results:")
    print("  - 3 ZIP files will be extracted and deleted")
    print("  - Files will be merged into 'test_data/merged/' folder")
    print("  - Extracted directories will be cleaned up")
    print("  - Only 'merged' folder will remain")

if __name__ == "__main__":
    main()

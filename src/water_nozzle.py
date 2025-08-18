# Import of the fisherman Library

from fisherman import (
    Project,
    OpenBoundary,
    OpenBoundaryDirection,
    OpenBoundaryType,
    Simulation,
    Settings,
    Part,
    ShapeType,
    FluidMaterial,
)

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fisherman.types import Cube, RotatoryMotion
from fisherman.enums import Unit

from include.materials import fluids
from include.auxiliary import get_sos


def get_user_input():

    delete_confirm = input("Are you sure you want to delete this simulation? (y/n)")

    if delete_confirm == "y" or delete_confirm == "yes":
        print("Deleting simulation.")
        return delete_confirm
    elif delete_confirm == "n" or delete_confirm == "no":
        print("Reconfigure.")
        return delete_confirm
    else:
        print("Enter y (yes) or n (no)!")
        get_user_input()

proj_name = "Tutorials"

sim_name = "Water Nozzle"
nozzle_name = "Water Slit Nozzle"
path_nozzle = r"static\project\water_slit_nozzle.stl"
path_nozzle_inlet = r"static\project\inlet_water_slit_nozzle.stl"

nozzle_diamater = 0.0009
d_particle = round(nozzle_diamater/10, 7)

# specify parameters of simulation
inlet_velocity = 0.21039
speed_of_sound = get_sos(inlet_velocity, d_particle, 3e5, 999) # capped to 1000m/s
end_time = 2
output_number = 400
output_time_interval = end_time / output_number

# extend class
class Project_ext(Project):

    @property
    def name(self):
        return self.data.get("name")

    def print_summary(self):
        print(f"Project '{self.name}' with ID {self.id}")

# get current projects, as list of dicts
projects = Project_ext.get(name=proj_name)

def get_project_id(name, projects):

    for p in projects:
        p.print_summary()

        if p.name == name:
            print(f"\n-- Using '{name}' Project -- \n")
            return p.id

def main():
    id = get_project_id(proj_name, projects)

    project = Project(id)

    ## altenatively create a new project called
    # project = Project()
    # project.add(name="moving-square-tutorial", description="Tutorial example of a dambreak")

    # add simulation to the project
    simulations = project.get_simulations()

    sim_found = False

    if simulations:
        print("\n Current simulations:\n")
    for s in simulations:
        print(s)

        if s.data.get("name") == sim_name:
            print(f"Simulation name is already used. Defaulting to using existing simulation: '{sim_name}'\n")
            simulation = s
            sim_found = True

            del_sim = get_user_input()

            if del_sim == "y" or del_sim == "yes":
                s.delete()
                print("Simulation has been deleted.")
                sim_found = False
            elif (del_sim == "n" or del_sim == "no"):
                sim_found = True
                
    if sim_found == False:
        print(f"\nCreating Simulation: {sim_name}\n in Project:{proj_name}\n")
        simulation = project.add_simulation(name=sim_name)

        # Preprocessor Setup

        # # housing
        # cube = simulation.add_wall_boundary("housing", shape_type=ShapeType.CUBE, flip_normals=True)
        # cube.scale({"x": 2, "y": 0.5, "z": 1})
        # cube.translate([1, 0.25, 0.5])

        # object
        cube_motion = RotatoryMotion(name="spin cube", vel=6.28, axis=[1, 0, 1], center=[0.95, 0.25, 0.25])

        cube = simulation.add_wall_boundary("cube", shape_type=ShapeType.CUBE)
        cube.scale({"x": 0.2, "y": 0.2, "z": 0.2})
        cube.translate([0.95, 0.25, 0.25])
        cube.add_motion(cube_motion)

        # nozzle
        nozzle = simulation.add_wall_boundary(name=nozzle_name, filepath=path_nozzle, unit=Unit.MILLIMETER)
        nozzle.scale([1, 1, 1])
        # 70.65, 269.25, -181.98 | [0.07065, 0.26925, -0.18198]
        nozzle.translate([0.79, 0.52, 0.3])
        nozzle.rotate([0, 45, 270])

        # nozzle inlet
        boundary_condition_inlet = OpenBoundary(
            io_direction=OpenBoundaryDirection.INLET,
            io_type=OpenBoundaryType.RIEMANN_VELOCITY,
            velocity=inlet_velocity, # m/s,
            )

        nozzle_inlet = simulation.add_open_boundary(name=f"{nozzle_name}_inlet", filepath=path_nozzle_inlet, open_boundary=boundary_condition_inlet, unit=Unit.MILLIMETER)
        nozzle_inlet.scale([1, 1, 1])
        nozzle_inlet.translate([0.782249966093799, 0.530995440667734, 0.30783988588701355])
        nozzle_inlet.rotate([0, 45, 270])

        # nozzle outlet
        boundary_condition_outlet = OpenBoundary(
            io_direction=OpenBoundaryDirection.OUTLET,
            io_type=OpenBoundaryType.RIEMANN_PRESSURE,
            pressure=0,
            )

        nozzle_inlet = simulation.add_open_boundary(name=f"{nozzle_name}_outlet", filepath=path_nozzle_inlet, open_boundary=boundary_condition_outlet, unit=Unit.METER)
        nozzle_inlet.scale([0.0015, 0.0015, 0.0015])
        nozzle_inlet.translate([70.71225, 280.251, -182.0095])
        nozzle_inlet.rotate([0, 45, 270])

        # refinement zone
        # 0.85, 0.25, 0.35 for inlet center
        refinement_cube_mesh = Cube(
            dimensions=[0.25, 0.25, 0.25],
            translation=[0.85, 0.25, 0.35],
        )
        refinement_cube = simulation.add_refinement_zone(name="refinement_cube", mesh=refinement_cube_mesh)

        # add deletion zone
        # 0.7, -0.1, 0 | 1.25, 0.6, 0.5
        # deletion = simulation.add_wall_boundary()

        # add disc sensor
        # 0.75, 0.25, 0.45 | 1, 0, -1 | 0.05

    # water
    fluid_params = fluids["Water_20C_3bar"]
    air = simulation.use_material(
        FluidMaterial(**fluid_params, speed_of_sound=speed_of_sound))


    # simulation settings
    settings = Settings(
        end_time=end_time,
        gravity=[0, 0, -9.81],
        particle_diameter=d_particle,
        output_time_interval=output_time_interval,
    )

    simulation.update_settings(settings)

    # simulate

    # simulation.start("gpu")

if __name__ == "__main__":
    main()
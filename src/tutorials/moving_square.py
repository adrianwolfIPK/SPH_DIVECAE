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

# specify parameters of simulation
speed_of_sound = 30
end_time = 1.5
output_number = 101
output_time_interval = end_time / output_number

# extend class
class Project_ext(Project):

    @property
    def name(self):
        return self.data.get("name")

    def print_summary(self):
        print(f"Project '{self.name}' with ID {self.id}")

# get current projects, as list of dicts
projects = Project_ext.get(name="Tutorials")

def get_project_id(name, projects):

    for p in projects:
        p.print_summary()

        if p.name == name:
            print(f"-- Using '{name}' Project -- \n")
            return p.id
    
id = get_project_id("Tutorials", projects)

project = Project(id)

## altenatively create a new project called
# project = Project()
# project.add(name="moving-square-tutorial", description="Tutorial example of a dambreak")

# add simulation to the project
simulation = project.add_simulation(name="Moving Square")

# Preprocessor Setup

# add cube
cube = simulation.add_wall_boundary("housing", shape_type=ShapeType.CUBE, flip_normals=True)
cube.scale({"x": 1.2, "y": 0.6, "z": 0.5})

# add moving square
moving_square = simulation.add_wall_boundary("moving_square", shape_type=ShapeType.CUBE)
moving_square.translate({"x": -0.5, "y": 0, "z": 0})
moving_square.scale({"x": 0.1, "y": 0.1, "z": 1})

# add movement
moving_square.add_translatory_movement(name="movement", start=0, end=1, acc=2)

# add fluid
fluid_name = "Water 20°C"
water = simulation.use_material(
    FluidMaterial(
        name="Water 20°C",
        density=998.21,
        viscosity=0.000001003,
        isentropic_exp=7,
        speed_of_sound=speed_of_sound,
    )
)

# water filling
filling_height = -.125

seed_point = [-.01, -.01, filling_height - 0.025]
water_filling = simulation.add_filling(name="filling", seed_point=seed_point, material=water)
water_filling.set_height(filling_height)

# simulation settings
settings = Settings(
    end_time=end_time,
    gravity=[0, 0, -9.81],
    particle_diameter=0.009,
    output_time_interval=output_time_interval,
)


simulation.update_settings(settings)

# simulate

simulation.start("gpu")
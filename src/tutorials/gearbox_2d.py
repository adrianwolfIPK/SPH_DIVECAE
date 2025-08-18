import sys
sys.path.append(r"C:\Users\adr61871\Desktop\FUSION_DIVE_Automatization")

from fisherman import Cube, Cylinder, Project, RefinementZone, Simulation, config
from include.fishkop import fishify

options = {
    "project_name" : "Tutorials", #set your project name (if non existent it will be created for you)
    "simulation_name" : "Gearbox_2d", # set simulation name
    "start_time" : 0.25, # s ... time to accelerate this system to constant velocity
    "end_time" : 2, # s ... end time of simulation
    "output_time_interval" : 0.01, # s ... output every x seconds
    "particle_diameter" : 2.0e-3, # m i.e., particle size
    "gravity" : [0, 0, 9.81], # gravity vector points towards negative z-direction (check your coordinate system !!!)
    "calc_domain_min" : [-0.063, -0.18675, -0.096], # minimum and maximum coordinates of box-shaped calculatory domain...
    "calc_domain_max" : [0.006, 0.09475, 0.096], # ... set it a few mm larger than the casing for the FZG tutorial
}

number_of_teeth_pinion = 16
number_of_teeth_cog = 24
rotation_speed_pinion = 56.55 # rad/s
rotation_speed_cog = - number_of_teeth_pinion/number_of_teeth_cog * rotation_speed_pinion
geometry_base = "static/" # path to directory with stl files "./" means the current directory

parts = {
    "Case": {
        "stl": f"{geometry_base}Case.stl", # note the usage of python f-strings to interpolate the directory name in the path
        },

    "Cog": {
        "stl": f"{geometry_base}Cog.stl",
        "omega": rotation_speed_cog,
        "center": [0., -0.0915, 0.],
        "axis": [1, 0, 0],
        },

    "pinion": {
        "stl": f"{geometry_base}pinion.stl",
        "omega": rotation_speed_pinion,
        "center": [0., 0.0, 0.0],
        "axis": [1, 0, 0],
        },
    }

#phase definitions
oil_level = -0.0322   # m parameter defining the filling height
fillings={ # dictionary defining the fillings
    "Liquid": {
        "material" : "Oil",
        "density" : 853.9, # kg/m^3
        "viscosity" : 0.000044585, # m^2/s
        "isentropic exponent" : 7,
        "speed of sound" : 30, # m/s
        "seed point" : [-0.01, 0.0, -0.08],
        "level" : oil_level,
        },
    }

simulation = fishify(options, parts, fillings)  

# Add the refinement zones
cogmesh = Cylinder(radius=0.05,height=0.075, axis=[1, 0, 0],translation=[-0.028673, -0.090813, -0.000286])
pinionmesh = Cylinder(radius=0.05,height=0.075, axis=[1, 0, 0],translation=[-0.029599, 0, 0.001327])
simulation.add_refinement_zone(name="refinement-pinion", mesh=pinionmesh)
simulation.add_refinement_zone(name="refinement-cog", mesh=cogmesh)

simulation.start(cores='gpu', confirm=False)
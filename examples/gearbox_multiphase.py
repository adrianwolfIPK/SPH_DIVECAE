## load functionality from fisherman and load numpy

from include.auxiliary import create_or_delete_sim
from include.fishkop import fishify
from fisherman import Project

options = { 
    "project_name" : "Tutorials", #set your project name (if non existent it will be created for you)
    "simulation_name" : "Tut2c_FZG_fisherman", # set simulation name
    "start_time" : 0.25, # s ... time to accelerate this system to constant velocity
    "end_time" : 2, # s ... end time of simulation
    "output_time_interval" : 0.01, #s ... output every x seconds
    "particle_diameter" : 2.0e-3, #m i.e., particle size 
    "gravity" : [0, 0, -9.81], #gravity vector points towards negative z-direction (check your coordinate system !!!)
    "calc_domain_min" : [-0.063, -0.18675, -0.096], #minimum and maximum coordinates of box-shaped calculatory domain...
    "calc_domain_max" : [0.006, 0.09475, 0.096], #...set it a few mm larger than the casing for the FZG tutorial 
}

create_or_delete_sim(options["project_name"], options["simulation_name"])

number_of_teeth_pinion = 16 
number_of_teeth_cog = 24
rotation_speed_pinion = 56.55 #rad/s
rotation_speed_cog = - number_of_teeth_pinion/number_of_teeth_cog * rotation_speed_pinion

geometry_base = "FZG_walls/" #path to directory with stl files "./" means the current directory
parts = {
    "Case": {
        "stl": f"{geometry_base}Case.stl",
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
        "material" : "Oil Custom",
        "density" : 853.9, #kg/m^3
        "viscosity" : 0.000044585, #m^2/s
        "isentropic exponent" : 7,
        "speed of sound" : 30, #m/s
        "seed point" : [-0.01, 0.0, -0.08],
        "level" : oil_level,
        },
    "Gas": {
         "material": "Air Custom",
         "density" : 1.225, #kg/m^3
         "viscosity" : 0.000015, #m^2/s
         "isentropic exponent" : 1.4,
         "speed of sound" : 60, #m/s
         "seed point" : [-0.01, 0.0, +0.08],
         "level" : oil_level,
         "second_material":True,
         },
    }

simulation = fishify(options, parts, fillings)

simulation.start("gpu")
# -*- coding: utf-8 -*-

import sys
sys.path.append(r"C:\Users\adr61871\Desktop\FUSION_DIVE_Automatization")

from include.fishkop import fishify
import numpy as np


# basic settings
options = {
    "project_name" : "Tutorials", #set your project name (if non existent it will be created for you)
    "simulation_name" : "Hourglass Surfacetension", # set simulation name 
    #"start_time" : 0.0,  # s ... time to accelerate this system to constant velocity
    "end_time" : 10.0,  # s ... end time of simulation
    "output_time_interval" : 0.05,  #s ... output every x seconds
    "particle_diameter" : 0.0003,  #m i.e., particle size
    "gravity" : [0.0, 0, -9.81], #gravity vector points towards negative z-direction (check your coordinate system !!!)
    "calc_domain_max" : [0.02, 0.02, 0.04],
    "calc_domain_min" : [-0.02, -0.02, -0.04], #minimum and maximum coordinates of box-shaped calculatory domian.
    "speed_of_sound" : 9.0,# 1.2*10*np.sqrt(70.0E-3/1000/0.5), #m/s
    "shifting": "pro",
    "rainbow" : True,
    "sigma":  70.0E-3, #N/m
    "surface_tension_parameter": 0.001,
    "dynamic_time_step": False,
}

geometry_base = "static/hourglass/" #path to directory with stl files "./" means the current directory
parts = {
    "hourglass": {
        "stl": f"{geometry_base}hourglass.stl",
        "contact_angle": 70.0,
        "flip":True,
    },
}

fillings={
    "Water": {
        "material": "custom oil",
        "density": 1000.0, #kg/m^3
        "viscosity":  0.000001, #m2/s
        "isentropic exponent": "liquid",
        "stl": f"{geometry_base}filling.stl",
        "seed point": [0.0, 0.0, 0.003],
        },
    }

simulation = fishify(options, parts, fillings)

# alerts
credit_alert = simulation.set_alert(
        action="stop",
        value_type="credits",
        threshold=200,
    )

simulation.start(cores='gpu', confirm=False)
    
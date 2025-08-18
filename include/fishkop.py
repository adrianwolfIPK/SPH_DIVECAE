# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 11:13:34 2021

@author: GeorgMensah

updated: 28/05/2024 by Benjamin Legrady
- adapted to new motion definition (Rotatory & Translatory Motion)
- adapted periodic boundary condition definition
- adapted slip & motion boundary condition
- extension of fishkop to use add sensors
- extension of fishkop to evaluate simulations through eval function
- includes update and improvement on ipx4 automation
- includes bugfixes for bearing automation
- contains possibility to control bearing automation numbering direction
- bugfixes on thermal boundary conditions
- includes possibility to use complex materials (non-newtonian & temperature dependent)
- added keyword "follow" on parts that takes list of components to consider as the leading motion
"""
import os
from copy import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load functionality from fisherman and load numpy
# from xmlrpc.client import Boolean
from fisherman import (
    Circle,
    Colors,
    ComplexMesh,
    Component,
    ComponentType,
    Cube,
    Cylinder,
    Datatable,
    FluidMaterial,
    MaterialViscosityKind,
    OpenBoundary,
    OpenBoundaryDirection,
    OpenBoundaryType,
    Project,
    Rectangle,
    RotationPeriodicBoundary,
    RotatoryMotion,
    Settings,
    ShapeType,
    Shifting,
    ThermoBoundaryCondition,
    TranslatoryMotion,
    Unit,
)
from scipy.spatial.transform import Rotation as R
# from stl import mesh

# import logging
# from fisherman import config
# config.logger.setLevel(logging.DEBUG) # debug logging
# config.Config.DEV = True
# config.Config.BASE_URL = "https://app.production.divesolutions.de/api"
# config.Config.BASE_URL = "https://app.preview.divesolutions.de/api"
# config.Config.BASE_URL = "https://app.protoa.sandbox.divesolutions.de/api"
# config.Config.BASE_URL = "https://app.protoa.sandbox.divesolutions.de/api"
# config.Config.BASE_URL = "https://app.protoheavy.sandbox.divesolutions.de/api"


# color wheel
def color_parts(parts):
    color_wheel = [
        Colors.GREY,
        Colors.BRIGHT_BLUE,
        Colors.BRIGHT_RED,
        Colors.BRIGHT_GREEN,
        Colors.YELLOW,
        Colors.ORANGE,
        Colors.PURPLE,
        Colors.DARK_BLUE,
        Colors.DARK_RED,
        Colors.DARK_GREEN,
    ]
    number_of_colors = len(color_wheel)
    idx = 0
    for part in parts:
        parts[part]["color"] = color_wheel[idx % number_of_colors]
        idx += 1

    # return parts


def automated_refinement(part: dict, dr: float, unit_scaling: float):
    # collect properties to easiliy anticipate the object's location and dimensions
    print(f"Automated refinement zone creation starts.")
    stl = part["stl"]
    if "scale" in part:
        scale = part["scale"]
        if type(scale) is not list:
            scale = [scale] * 3
    else:
        scale = [1, 1, 1]

    if "rotate" in part:
        rotate = part["rotate"]
    else:
        rotate = [0, 0, 0]

    if "translate" in part:
        translate = part["translate"]
    else:
        translate = [0, 0, 0]

    # assess bounding box center
    obj = mesh.Mesh.from_file(stl)
    # center_bound = np.zeros(3)
    center_bound = 0.5 * (obj.max_ + obj.min_) * unit_scaling

    dr_dist = [
        5 * dr / unit_scaling / s for s in scale
    ]  # 5 x particle diameter distance to boundary

    bounding_points_original = [
        [x * unit_scaling, y * unit_scaling, z * unit_scaling]
        for x in [obj.min_[0].item() - dr_dist[0], obj.max_[0].item() + dr_dist[0]]
        for y in [obj.min_[1].item() - dr_dist[1], obj.max_[1].item() + dr_dist[1]]
        for z in [obj.min_[2].item() - dr_dist[2], obj.max_[2].item() + dr_dist[2]]
    ]

    # scale object
    bounding_points_dimension = [
        [(a - b) * s + b for (a, b, s) in zip(bp, center_bound, scale)]
        for bp in bounding_points_original
    ]

    # translate object
    center_bound = [cb + t for (cb, t) in zip(center_bound, translate)]
    bounding_points_translate = [
        [a + t for (a, t) in zip(bp, translate)] for bp in bounding_points_dimension
    ]

    if "axis" in part:
        # cylindrical refinement zone
        if "center" in part:
            center = part["center"]
        else:
            center = center_bound

        # rotate object
        r = R.from_euler("xyz", rotate, degrees=True)
        bounding_points_tmp = [
            r.apply([a - c for (a, c) in zip(bp, center_bound)])
            for bp in bounding_points_translate
        ]  # rotate around centre point
        bounding_points = [
            [a + c for (a, c) in zip(bp, center_bound)] for bp in bounding_points_tmp
        ]  # move back to centre

        # axis, centre, radius and height determination of cylinder
        axis = part["axis"]
        axis = [a / np.linalg.norm(axis) for a in axis]
        c = np.array(center)
        ax = np.array(axis)
        # maximum radius size
        radius = max(
            [
                np.linalg.norm(np.cross(ax, c - np.array(bp))) / np.linalg.norm(ax)
                for bp in bounding_points
            ]
        ) / np.sqrt(2)

        projected_points = [
            (np.array(bp) - c) / (np.dot(ax, ax)) * ax + c for bp in bounding_points
        ]
        projected_points_min = [
            min(pp[0] for pp in projected_points),
            min(pp[1] for pp in projected_points),
            min(pp[2] for pp in projected_points),
        ]
        projected_points_max = [
            max(pp[0] for pp in projected_points),
            max(pp[1] for pp in projected_points),
            max(pp[2] for pp in projected_points),
        ]
        center_zone = [
            0.5 * (a + b) for (a, b) in zip(projected_points_min, projected_points_max)
        ]
        height = np.linalg.norm(
            [(a - b) for (a, b) in zip(projected_points_min, projected_points_max)]
        )

        zone = {
            "radius": radius,
            "height": height,
            "axis": part["axis"],
            "center": center_zone,
        }
    else:
        # cubical refinemet zone
        dimension = [
            max(pp[0] for pp in bounding_points_translate)
            - min(pp[0] for pp in bounding_points_translate),
            max(pp[1] for pp in bounding_points_translate)
            - min(pp[1] for pp in bounding_points_translate),
            max(pp[2] for pp in bounding_points_translate)
            - min(pp[2] for pp in bounding_points_translate),
        ]
        zone = {"dimension": dimension, "rotation": rotate, "center": center_bound}

    return zone


# features for planetary gearings
def listify(array_like):  # auxiliary function to convert numpy arrays
    return [a for a in array_like]


def planetize(parts):
    """
        planetize(parts)

    Modify entries of `parts` for planetary gearings, i.e., create copies of the planets
    put them to the correct positions, and compute the correct rotation velocities for the various components.
    Note that currently only single stage planetary gearings are supported.
    """
    to_be_added = {}
    to_be_deleted = []
    for name, part in parts.items():
        if "planets" in part:
            # Willis equation for single stage planetary gear
            # TODO: multi-stage
            # degree of freedom index:
            # sun: 0, planet: 1, carrier: 2, ring: 3

            A = np.zeros((4, 4))
            y = np.zeros(4)

            teeth_planet = part["teeth"]
            teeth_sun = parts[part["sun"]]["teeth"]
            teeth_ring = parts[part["ring"]]["teeth"]
            A[0, 0] = teeth_sun
            A[0, 1] = teeth_planet
            A[0, 2] = -(teeth_sun + teeth_planet)
            A[1, 0] = teeth_sun
            A[1, 3] = teeth_ring
            A[1, 2] = -(teeth_sun + teeth_ring)
            counter = 0
            if "omega" in part:
                A[counter + 2, 1] = 1
                y[counter + 2] = part["omega"]
                counter += 1
            if "omega" in parts[part["sun"]]:
                A[counter + 2, 0] = 1
                y[counter + 2] = parts[part["sun"]]["omega"]
                counter += 1
            if "omega" in parts[part["carrier"]]:
                A[counter + 2, 2] = 1
                y[counter + 2] = parts[part["carrier"]]["omega"]
                counter += 1
            if "omega" in parts[part["ring"]]:
                A[counter + 2, 3] = 1
                y[counter + 2] = parts[part["ring"]]["omega"]
                counter += 1

            if counter > 2:
                print(
                    "Error, too many rotations specified in planetary gearing system!"
                )
            elif counter < 2:
                print(
                    f"You specified {counter} rotations for the planetray gearing system but 2 are required."
                )

            # compute rotational speeds
            x = np.linalg.solve(A, y)

            parts[part["sun"]]["omega"] = x[0]
            parts[part["carrier"]]["omega"] = x[2]
            parts[part["ring"]]["omega"] = x[3]
            part["omega"] = x[1]

            number_of_copies = part["planets"]
            leader = part["carrier"]
            part["axis_leader"] = parts[leader]["axis"]
            part["omega_leader"] = parts[leader]["omega"]
            part["center_leader"] = parts[leader]["center"]
            part["omega_follower"] = part["omega"] - parts[leader]["omega"]

            del part["omega"]
            del part["carrier"]
            del part["planets"]

            axis = np.array(part["axis_leader"])
            axis = axis / np.linalg.norm(axis)
            center_leader = np.array(part["center_leader"])
            center = np.array(part["center"])
            radius = center - center_leader
            footpoint = np.dot(radius, axis) * axis + center_leader
            radius = center - footpoint
            # rr = np.linalg.norm(radius)
            perp = np.cross(radius, axis)
            for i in range(number_of_copies):
                new_part = part.copy()
                phi = i * 2 * np.pi / number_of_copies
                translate_vector = (
                    radius * np.cos(phi) + perp * np.sin(phi) + footpoint - center
                )
                new_part["center"] = listify(center + translate_vector)
                new_part["translate"] = listify(translate_vector)
                if "contact_angle" in part:
                    new_part["contact_angle"] = part["contact_angle"]
                to_be_added[f"{name}_{i+1}"] = new_part
            to_be_deleted.append(name)

        if "rollers" in part:
            # Bearing kinematics
            # TODO: multi-stage
            # TODO: make computation axes independent
            # degree of freedom index:
            # inner_ring: 0, roller: 1, cage: 2, outer_ring: 3
            # nomenclature:
            #   alpha: contact angle
            #   dp: pitch diameter
            #   dr: roller diameter

            # Determine who is moving
            axis = [0, 0, 0]
            center = [0, 0, 0]
            omega_in = 0
            omega_out = 0

            if part["inner_ring"] in parts:
                if "omega" in parts[part["inner_ring"]]:
                    omega_in = parts[part["inner_ring"]]["omega"]
                    axis = parts[part["inner_ring"]]["axis"]
                    center = parts[part["inner_ring"]]["center"]
                elif "omega_follower" in parts[part["inner_ring"]]:
                    omega_in = parts[part["inner_ring"]]["omega_follower"]
                    axis = parts[part["inner_ring"]]["axis_follower"]
                    center = parts[part["inner_ring"]]["center_follower"]
                else:
                    omega_in = 0
            else:
                print(f"Part {part['inner_ring']} not defined. Omega_in defined as 0.")
            if part["outer_ring"] in parts:
                if "omega" in parts[part["outer_ring"]]:
                    omega_out = parts[part["outer_ring"]]["omega"]
                    axis = parts[part["outer_ring"]]["axis"]
                    center = parts[part["outer_ring"]]["center"]
                elif "omega_follower" in parts[part["outer_ring"]]:
                    omega_out = parts[part["outer_ring"]]["omega_follower"]
                    axis = parts[part["outer_ring"]]["axis_follower"]
                    center = parts[part["outer_ring"]]["center_follower"]
                else:
                    omega_out = 0
            else:
                print(f"Part {part['outer_ring']} not defined. Omega_out defined as 0.")
            if (omega_in == 0 and omega_out == 0) or (axis == [0, 0, 0]):
                print(
                    f"Error, you haven't specified the inner and outer rings for {name}."
                )

            # Get or compute contact angle
            alpha = 0
            if "contacting_angle" in part:
                alpha = part["contacting_angle"]
                if (10 > alpha) or (alpha > 30):
                    print(
                        f"The specified contacting angle between roller and races {round(alpha,1)} is outside of the typical 10 - 30 deg range."
                    )
                alpha = np.radians(alpha)
            else:
                vector_1 = np.asarray(part["axis"])
                vector_2 = np.asarray(axis)
                unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
                unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
                dot_product = np.dot(unit_vector_1, unit_vector_2)
                alpha = np.arccos(dot_product)

            # Get or compute pitch diameter
            if "pitch_diameter" in part:
                dp = part["pitch_diameter"]
            else:
                vector_1 = np.asarray(center)
                vector_2 = np.asarray(axis)
                vector_3 = np.asarray(part["center"])
                dp = (
                    np.linalg.norm(np.cross(vector_2, vector_1 - vector_3))
                    / np.linalg.norm(vector_2)
                    * 2
                )

            dr = part["roller_element_diameter"]

            # compute rotational speeds
            omega_cage = 0.5 * (
                omega_in * (1 - dr / dp * np.cos(alpha))
                + omega_out * (1 + dr / dp * np.cos(alpha))
            )

            omega_roller = (
                0.5 * (omega_out - omega_in) * (dp / dr - dr / dp * np.cos(alpha) ** 2)
            )
            if "cage" in part:
                parts[part["cage"]]["omega"] = omega_cage
                parts[part["cage"]]["axis"] = axis
                parts[part["cage"]]["center"] = center
                del part["cage"]

            # leader = part["carrier"]
            part["axis_leader"] = axis
            part["omega_leader"] = omega_cage
            part["center_leader"] = center
            part["omega_follower"] = omega_roller

            number_of_copies = part["rollers"]
            # clean up

            del part["rollers"]
            del part["inner_ring"]
            del part["outer_ring"]
            del part["roller_element_diameter"]

            # assess bounding box center
            obj = mesh.Mesh.from_file(part["stl"])
            # center_bound = np.zeros(3)
            center_bound = 0.5 * (obj.max_ + obj.min_)

            # prepare for copies
            axis = np.array(part["axis_leader"])
            axis = axis / np.linalg.norm(axis)
            center_leader = np.array(part["center_leader"])
            center = np.array(part["center"])
            radius = center - center_leader
            radius_bound = center_bound - center_leader

            footpoint = np.dot(radius, axis) * axis + center_leader
            footpoint_bound = np.dot(radius_bound, axis) * axis + center_leader

            radius = center - footpoint
            radius_bound = center_bound - footpoint_bound

            rr = np.linalg.norm(radius)
            rr_bound = np.linalg.norm(radius_bound)

            perp = np.cross(
                radius, axis
            )  # changing cross product entries corrects order
            perp_bound = np.cross(
                radius_bound, axis
            )  # changing cross product entries corrects order

            for i in range(number_of_copies):
                new_part = part.copy()
                phi = i * 2 * np.pi / number_of_copies
                translate_vector = (
                    radius * np.cos(phi) + perp * np.sin(phi) + footpoint - center
                )
                translate_vector_bound = (
                    radius_bound * np.cos(phi)
                    + perp_bound * np.sin(phi)
                    + footpoint_bound
                    - center_bound
                )
                rotation_vector = -phi * np.array(axis)
                rotation = R.from_rotvec(rotation_vector)
                rotated_axis = rotation.apply(part["axis"])

                new_part["axis"] = listify(rotated_axis)
                new_part["center"] = listify(center + translate_vector)
                new_part["translate"] = listify(translate_vector_bound)
                new_part["rotate"] = listify([np.rad2deg(x) for x in rotation_vector])

                for key in [
                    "contact_angle",
                    "unit",
                    "temperature",
                    "adiabatic",
                    "heat_flux",
                    "htc",
                    "reference_temperature",
                    "flip",
                    "wireframe",
                    "color",
                    "visibility",
                    "follow",
                    "start_motion",
                    "velocity_boundary_condition",
                    "slip_wall",
                ]:
                    if key in part:
                        new_part[key] = part[key]
                to_be_added[f"{name}_{i+1}"] = new_part
            to_be_deleted.append(name)

    for name in to_be_deleted:
        del parts[name]
    for name, part in to_be_added.items():
        parts[name] = part

    return None


def link_rotations(parts):
    for name, part in parts.items():
        if "omega" in part and part["omega"] in parts:
            if "omega" in parts[part["omega"]]:
                part["omega"] = parts[part["omega"]][
                    "omega"
                ]  # TODO: check whether this is a number...
            else:
                print(
                    f"Error: part {name} is referencing {part['omega']} for its rotatonale speed. However, no speed is set for {part['omega']}."
                )

    for name, part in parts.items():
        if ("omega" in part) and np.abs(part["omega"]) < 3e-14:
            del parts[name]["omega"]

    return None


# %% main part
# auxiliary function to identify whether simulation is multiphase
def ismultiphase(parts, fillings):
    fluids = []
    for name, part in parts.items():
        if "fluid" in part:
            if part["fluid"] not in fluids:
                fluids.append(part["fluid"])
    for name, filling in fillings.items():
        if "material" in filling:
            if filling["material"] not in fluids:
                fluids.append(filling["material"])
    return len(fluids) > 1


def issurfacetension(options, parts, fillings):
    for name, filling in fillings.items():
        if "sigma" in filling:
            options["sigma"] = filling["sigma"]
            return True
    if "sigma" in options:
        return True  # TODO:
    else:
        return False


# auxiliary function to create inputs for rotatory movements


def insert2rotatory(part, options, kinematic_type=""):
    if kinematic_type != "":
        kinematic_type = "_" + kinematic_type
    inputs = {}
    if "name" in part:
        inputs["name"] = part["name"]  # TODO: type checks
    if "start_time" in part:
        inputs["start"] = part["start_time"]
    elif "start_time" in options:
        inputs["start"] = options[
            "start_time"
        ]  # check wehther at least there is a global option
    if "end_time" in part:
        inputs["end"] = part["end_time"]
    elif "end_time" in options:
        inputs["end"] = options[
            "end_time"
        ]  # check wehther at least there is a global option
    if "omega" + kinematic_type in part:
        inputs["vel"] = part["omega" + kinematic_type]
    if kinematic_type == "_follower":
        inputs["leader_id"] = part["dive-movement-leaders"]
    if "axis" + kinematic_type in part:
        inputs["axis"] = part["axis" + kinematic_type]
    elif "axis" in part:
        inputs["axis"] = part["axis"]
    if "center" + kinematic_type in part:
        inputs["center"] = part["center" + kinematic_type]
    elif "center" in part:
        inputs["center"] = part["center"]
    if "acc" in part:
        if type(part["acc"]) == float:
            inputs["acc"] = part["acc"]
        elif type(part["acc"]) == str:
            inputs["acceleration_csv"] = part["acc"]
    if "virtual_movement" in part:
        print(
            "Virtual Movements on parts will be deprected soon. Please add it as velocity_boundary_condition"
        )
        # inputs["virtual_movement"] = part["virtual_movement"]
    # TODO: leader id
    return inputs


def find_rotation_angles(start_vec, end_vec):

    start_unit = start_vec / np.linalg.norm(start_vec)
    end_unit = end_vec / np.linalg.norm(end_vec)
    rot_axis = np.cross(start_unit, end_unit)
    cos_angle = np.dot(start_unit, end_unit)
    sin_angle = np.linalg.norm(rot_axis)
    if sin_angle > 1e-6:
        rot_axis /= sin_angle
    else:
        if cos_angle > 0:
            return 0, 0, 0
        else:
            if abs(start_unit[0]) < abs(start_unit[1]) and abs(start_unit[0]) < abs(
                start_unit[2]
            ):
                rot_axis = np.array([1, 0, 0])
            else:
                rot_axis = np.array([0, 1, 0])

    rot_angle = np.arctan2(sin_angle, cos_angle)

    rot_quaternion = R.from_rotvec(rot_axis * rot_angle)

    euler_angles = rot_quaternion.as_euler("xyz", degrees=False)

    return euler_angles


class Ipx4Setup:

    def __init__(self, radius, gravity, num_points, center):

        self.radius = radius
        self.gravity = gravity
        self.num_points = num_points
        self.center = np.array(center)

        self.nozzle_direction = np.array(gravity) / np.linalg.norm(np.array(gravity))
        self.bow_axis = self._get_bow_axis()
        self.rotation_axis = np.cross(self.bow_axis, self.nozzle_direction)

    def _get_bow_axis(self):

        if self.gravity[0] == 0 and self.gravity[1] == 0:
            non_collinear_vector = np.array([1, 0, 0])
        else:
            non_collinear_vector = np.array([0, 0, 1])
        perpendicular_vector = np.cross(np.array(self.gravity), non_collinear_vector)

        return perpendicular_vector / np.linalg.norm(perpendicular_vector)

    def _rotate_vector(self, theta):
        return (
            self.nozzle_direction * np.cos(theta)
            + np.cross(self.bow_axis, self.nozzle_direction) * np.sin(theta)
            + self.bow_axis
            * np.dot(self.bow_axis, self.nozzle_direction)
            * (1 - np.cos(theta))
        )

    def calculate_position_and_normal(self):

        theta = np.deg2rad(180 / (self.num_points))

        v_rot_dir1 = [
            -self._rotate_vector(theta * i) * self.radius + self.center
            for i in range(self.num_points // 2 + 1)
        ]
        v_rot_dir2 = [
            -self._rotate_vector(-theta * i) * self.radius + self.center
            for i in range(1, self.num_points // 2 + 1)
        ]
        v_rot = v_rot_dir2 + v_rot_dir1
        normals = [self.center - v for v in v_rot]

        return v_rot, [n / np.linalg.norm(n) for n in normals]


def create_circle_surface_stl(
    normal_vector, translation_vector, name, radius=10, segments=32
):

    normal = np.array(normal_vector) / np.linalg.norm(normal_vector)

    orthogonal = np.cross(normal, [1, 0, 0])

    if np.linalg.norm(orthogonal) == 0:
        orthogonal = np.cross(normal, [0, 1, 0])
    orthogonal /= np.linalg.norm(orthogonal)

    binormal = np.cross(normal, orthogonal)

    vertices = [translation_vector]
    for segment in range(segments):
        angle = 2 * np.pi * segment / segments
        vertex = (
            translation_vector
            + radius * np.cos(angle) * orthogonal
            + radius * np.sin(angle) * binormal
        )
        vertices.append(vertex)

    faces = []
    for i in range(1, segments):
        faces.append([0, i, i + 1])
    faces.append([0, segments, 1])  #

    circle_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            circle_mesh.vectors[i][j] = vertices[f[j]]
        circle_mesh.normals[i] = normal

    create_directory_if_not_exists("./inlet")
    circle_mesh.save(f"./inlet/{name}")


def create_directory_if_not_exists(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)


def ipx4(options, parts):

    if "central_rotation_rpm" in options["mode"]:
        omega_leader = options["mode"]["central_rotation_rpm"] / 60 * 2 * np.pi
    else:
        omega_leader = 3 / 60 * 2 * np.pi
        print("Central rotation (central_rotation_rpm) assigned with 3 RPM.")
    if "swing_rpm" in options["mode"]:
        omega_follower = options["mode"]["swing_rpm"] / 60 * 2 * np.pi
    else:
        omega_follower = 12 / 60 * 2 * np.pi
        print("Central rotation (swing_rpm) assigned with 12 RPM.")
    if "swing_duration_s" in options["mode"]:
        t_swing = options["mode"]["swing_duration_s"]
    else:
        t_swing = 2
        print("A swing takes 2 seconds.")
    if "bow_radius_mm" in options["mode"]:
        transform_radius = options["mode"]["bow_radius_mm"] / 1000
    else:
        transform_radius = None
    if "volumeflow_lpmin" in options["mode"]:
        v_flux = options["mode"]["volumeflow_lpmin"] / 60 / 1000
    else:
        v_flux = None
    if "nozzle_diameter_mm" in options["mode"]:
        nozzle_diameter = options["mode"]["nozzle_diameter_mm"] / 1000
        inlet_diameter = options["mode"]["nozzle_diameter_mm"] / 1000
    else:
        nozzle_diameter = 0.0004
        inlet_diameter = 0.002
        print("The standard nozzle diameter of 0.4 mm was assumed.")
    if "nr_nozzles" in options["mode"]:
        nr_nozzles = options["mode"]["nr_nozzles"]
    else:
        nr_nozzles = None
    if "automated_center" in options["mode"]:
        automated_center = options["mode"]["automated_center"]
    else:
        automated_center = False
    if "nozzle_name" in options["mode"]:
        if options["mode"]["nozzle_name"] in parts:
            exclude_nozzle = options["mode"]["nozzle_name"]
        else:
            print(
                f"The name {options['mode']['nozzle_name']} for the nozzle does not exist in parts."
            )
    else:
        print(f"You must define the name of the nozzle ('nozzle_name') in parts.")

    if "nozzle_direction" in options["mode"]:
        nozzle_direction = options["mode"]["nozzle_direction"]
    else:
        nozzle_direction = [np.sign(g) for g in options["gravity"]]
        print(f"Attention: The nozzle direction is taken from the gravity vector.")

    if "fluid" in parts[options["mode"]["nozzle_name"]]:
        fluid = parts[options["mode"]["nozzle_name"]]["fluid"]
    else:
        fluid = "Water 20°C"
        print(f"Material of water at 20°C is assumed.")
    if "refined" in options["mode"]:
        if options["mode"]["refined"] == True:
            for name, part in parts.items():
                if name != exclude_nozzle:
                    part["automated_refinement"] = True

    gravity = options["gravity"]

    # Combine remaining STLs
    paths = [part["stl"] for name, part in parts.items() if name != exclude_nozzle]
    meshes = [mesh.Mesh.from_file(path) for path in paths]
    combined = mesh.Mesh(np.concatenate([m.data for m in meshes]))
    center = [x for x in (0.5 * (combined.max_ + combined.min_)).tolist()]
    transform_vector = center

    # central point of rotation
    for iter, g_comp in enumerate(gravity):
        if g_comp != 0:
            transform_vector[iter] = (
                -np.sign(min(0, gravity[iter])) * combined.min_[iter]
                + np.sign(max(0, gravity[iter])) * combined.max_[iter]
            )

    # largest radius of object
    max_radius = np.linalg.norm(combined.max_ - transform_vector)
    ideal_transform_radius = (np.floor(max_radius / 0.2) + 1) * 0.2

    # Look-up table for IPX class 4
    if abs(ideal_transform_radius - 0.2) < 0.001:
        ideal_nr_nozzle = 13
        ideal_v_flux = 0.84 / 60 / 1000
    elif abs(ideal_transform_radius - 0.4) < 0.001:
        ideal_nr_nozzle = 25
        ideal_v_flux = 1.8 / 60 / 1000
    elif abs(ideal_transform_radius - 0.6) < 0.001:
        ideal_nr_nozzle = 37
        ideal_v_flux = 2.6 / 60 / 1000
    elif abs(ideal_transform_radius - 0.8) < 0.001:
        ideal_nr_nozzle = 51
        ideal_v_flux = 3.5 / 60 / 1000
    elif abs(ideal_transform_radius - 1.0) < 0.001:
        ideal_nr_nozzle = 63
        ideal_v_flux = 4.3 / 60 / 1000
    elif abs(ideal_transform_radius - 1.2) < 0.001:
        ideal_nr_nozzle = 75
        ideal_v_flux = 5.3 / 60 / 1000
    elif abs(ideal_transform_radius - 1.4) < 0.001:
        ideal_nr_nozzle = 87
        ideal_v_flux = 6.1 / 60 / 1000
    elif abs(ideal_transform_radius - 1.6) < 0.001:
        ideal_nr_nozzle = 101
        ideal_v_flux = 7 / 60 / 1000

    if transform_radius is None:
        transform_radius = ideal_transform_radius

    if nr_nozzles is None:
        nr_nozzles = ideal_nr_nozzle

    if v_flux is None:
        v_flux = ideal_v_flux

    inlet_v = v_flux / (nozzle_diameter * nozzle_diameter / 4 * np.pi) / nr_nozzles

    if automated_center == True:
        transform_vector = [-t for t in transform_vector]
        for name, part in parts.items():
            if name != exclude_nozzle:
                part["translate"] = transform_vector
                center = [0, 0, 0]

    setup = Ipx4Setup(transform_radius, gravity, nr_nozzles, center)
    translate, normal = setup.calculate_position_and_normal()

    for i in range(len(translate)):
        name = f"inlet_{i}.stl"
        create_circle_surface_stl(
            list(normal[i]), list(translate[i]), name, inlet_diameter, 30
        )

    if automated_center == True:
        transform_vector = [-t for t in transform_vector]
        for name, part in parts.items():
            if name != exclude_nozzle:
                part["translate"] = transform_vector
                center = [0, 0, 0]

    nozzles = {
        f"inlet_{x}": {
            "stl": f"./inlet/inlet_{x}.stl",
            "omega_leader": omega_leader,
            "axis_leader": list(np.float64(setup.nozzle_direction)),
            "center_leader": list(np.float64(center)),
            "oscillation_follower": omega_follower,
            "oscillation_shift": 0.5,
            "axis_follower": list(np.float64(setup.rotation_axis)),
            "center_follower": list(np.float64(center)),
            "delta_t": t_swing,
            "inlet_velocity": inlet_v,
            "fluid": parts[exclude_nozzle]["fluid"],
        }
        for x in range(nr_nozzles)
    }

    parts.update(nozzles)

    diff = options["particle_diameter"] * 2.5
    calc_max = [x + transform_radius + diff for x in center]
    calc_min = [x - transform_radius - diff for x in center]

    for iter, g_comp in enumerate(gravity):
        if g_comp < 0:
            calc_min[iter] = center[iter] - diff
        if g_comp > 0:
            calc_max[iter] = center[iter] + diff

    options["calc_domain_min"] = calc_min
    options["calc_domain_max"] = calc_max

    if "Inlet" in parts:
        del parts["Inlet"]


def ipx5(options, parts):
    """
        ipx5(parts, name)

    Function to prepare an IPX5 test.
    """
    print("IPX5 mode activated.")
    # get parameter
    if "object_rotation" in options["mode"]:
        omega_leader = options["mode"]["object_rotation"] / 60 * 2 * np.pi
    else:
        omega_leader = 0
        print("Central rotation (object_rotation) assigned with 0 RPM.")
    if "oscillation" in options["mode"]:
        oscillation_time = options["mode"]["oscillation"]
    else:
        oscillation_time = 0.4
        print("A vertical motion takes 0.4 seconds.")
    if "vertical_motion_range" in options["mode"]:
        vertical_motion_range = options["mode"]["vertical_motion_range"]
    else:
        vertical_motion_range = 0.5
        print("Vertical motion range (vertical_motion_range) assigned with 0.5 m.")

    if "volumeflow_lpmin" in options["mode"]:
        v_flux = options["mode"]["volumeflow_lpmin"] / 60 / 1000
    else:
        v_flux = None
    if "nozzle_diameter_mm" in options["mode"]:
        nozzle_diameter = options["mode"]["nozzle_diameter_mm"] / 1000
    else:
        nozzle_diameter = 0.0063
        print("The standard nozzle diameter of 0.4 mm was assumed.")

    if "nozzle_center" in options["mode"]:
        nozzle_center = options["mode"]["nozzle_center"]
    else:
        nozzle_center = False
    if "nozzle_direction" in options["mode"]:
        nozzle_direction = options["mode"]["nozzle_direction"]
    else:
        nozzle_direction = [np.sign(g) for g in options["gravity"]]
        print(f"Attention: The nozzle direction is taken from the gravity vector.")
    if "nozzle_name" in options["mode"]:
        if options["mode"]["nozzle_name"] in parts:
            exclude_nozzle = options["mode"]["nozzle_name"]
        else:
            print(
                f"The name {options['mode']['nozzle_name']} for the nozzle does not exist in parts."
            )
    else:
        print(f"You must define the name of the nozzle ('nozzle_name') in parts.")

    if "fluid" in parts[options["mode"]["nozzle_name"]]:
        fluid = parts[options["mode"]["nozzle_name"]]["fluid"]
    else:
        fluid = "Water 20°C"
        print(f"Material of water at 20°C is assumed.")
    if "refined" in options["mode"]:
        if options["mode"]["refined"] == True:
            for name, part in parts.items():
                if name != exclude_nozzle:
                    part["automated_refinement"] = True

    gravity = options["gravity"]

    # Combine remaining STLs
    paths = [part["stl"] for name, part in parts.items() if name != exclude_nozzle]
    meshes = [mesh.Mesh.from_file(path) for path in paths]
    combined = mesh.Mesh(np.concatenate([m.data for m in meshes]))

    # get unknown parameter if nothing is provided

    # center of rotation
    center = [x for x in (0.5 * (combined.max_ + combined.min_)).tolist()]

    inlet_v = v_flux / (nozzle_diameter * nozzle_diameter / 4 * np.pi)

    nozzle_relocation_vector_center = nozzle_center

    rotation_radius = np.linalg.norm(
        np.cross(
            np.array(gravity),
            np.array(center) - np.array(nozzle_relocation_vector_center),
        )
    ) / np.linalg.norm(gravity)

    # inlet property calculation
    rotation_center = center

    # rotation angle calculation
    vector_1 = [0, -1, 0]
    vector_2 = nozzle_direction
    rot_vec = np.cross(vector_1, vector_2) / np.linalg.norm(
        np.cross(vector_1, vector_2)
    )
    r = R.from_rotvec(rot_vec)
    center_rotation = r.as_euler("xyz", degrees=True)

    nozzle_rotationangle_center = [x.item() for x in center_rotation]

    # assign properties to inlets
    # start with first inlet
    nozzle_direction
    parts[exclude_nozzle]["translate"] = nozzle_relocation_vector_center
    parts[exclude_nozzle]["rotate"] = nozzle_rotationangle_center
    parts[exclude_nozzle]["scale"] = [nozzle_diameter * 1000 for x in [0] * 3]
    parts[exclude_nozzle]["omega"] = omega_leader
    parts[exclude_nozzle]["axis"] = gravity
    parts[exclude_nozzle]["center"] = rotation_center
    parts[exclude_nozzle]["oscillation_translation"] = (
        vertical_motion_range / oscillation_time
    )
    parts[exclude_nozzle]["axis_translation"] = [-g for g in gravity]
    parts[exclude_nozzle]["delta_t"] = oscillation_time
    parts[exclude_nozzle]["inlet_velocity"] = inlet_v

    # set calculation domain
    diff = options["particle_diameter"] * 2.5
    calc_max = [
        max(
            x.item(),
            y + diff + nozzle_diameter,
            y + z + diff + nozzle_diameter,
            [x.item(), rotation_radius + diff + nozzle_diameter + c][g == 0],
        )
        for (x, y, z, g, c) in zip(
            combined.max_,
            nozzle_relocation_vector_center,
            [-g * vertical_motion_range / 9.81 for g in gravity],
            gravity,
            center,
        )
    ]
    calc_min = [
        min(
            x.item(),
            -y - diff - nozzle_diameter,
            -y - z - diff - nozzle_diameter,
            [x.item(), -rotation_radius - diff - nozzle_diameter + c][g == 0],
        )
        for (x, y, z, g, c) in zip(
            combined.min_,
            nozzle_relocation_vector_center,
            [-g / 9.81 * vertical_motion_range for g in gravity],
            gravity,
            center,
        )
    ]
    options["calc_domain_min"] = [c for c in calc_min]
    options["calc_domain_max"] = [c for c in calc_max]


def mode_selection(options, parts):
    """
        mode_selection(parts)

    Function to preselect the simulation mode for e.g. IPX tests.
    Note that currently only IPX54 is supported.
    """
    if "mode" in options:
        if options["mode"]["type"] == "ipx4":
            ipx4(options, parts)
        if options["mode"]["type"] == "ipx5":
            ipx5(options, parts)
        else:
            print(f"Unknown type {options['mode']['type']}.")


# wrapper to fisherman
def fishify(options, parts, fillings={}, refinement_zones={}, sensors={}):
    mode_selection(options, parts)
    planetize(parts)
    link_rotations(parts)
    options["multiphase"] = ismultiphase(parts, fillings)
    options["surface_tension"] = issurfacetension(options, parts, fillings)

    # sanity checks
    assert options["end_time"] > 0
    if "start_time" in options:
        assert (
            options["start_time"] >= 0 and options["start_time"] <= options["end_time"]
        )
    # set shifting method
    if "shifting" not in options:
        options["shifting"] = "pro"
    # TODO: check whether isinstance str if no it might already be an enumeration type
    options["shifting"] = options["shifting"].strip().lower()
    # choose the correct enumeration type
    if options["shifting"] == "pro":
        options["shifting"] = Shifting.PRO
    elif options["shifting"] == "legacy":
        options["shifting"] = Shifting.LEGACY
    elif options["shifting"] == "minimum":
        options["shifting"] = Shifting.MINIMUM
    else:
        print(
            f"Warning: Shifting method '{options['shifting']}' not defined, defaulting to shifting = 'minimum'."
        )
        options["shifting"] = Shifting.MINIMUM

    #######################################
    # communication with cloud starts here
    ########################################
    # check whether project name is already existing if not create one.

    project = Project()
    try:
        project = Project.get(name=options["project_name"])[0]
    except:
        project.add(name=options["project_name"])

    # same with  simulation name
    try:
        simulation = project.get_simulations(name=options["simulation_name"])[0]
    except:
        simulation = project.add_simulation(
            name=options["simulation_name"],
        )

    # TODO: create default values if an option is missing
    # like
    if (
        "start_time" not in options
    ):  # should be redundant once translatory and rotatory input functions are ready
        options["start_time"] = 0.0

    # wrap settings in extra dictionary in order to make a call using the unpack operator. This is necessary, as some options are not strictly settings as defined by the fisherman API.
    settings = {}
    for key in [
        "end_time",
        "gravity",
        "multiphase",
        "particle_diameter",
        "output_time_interval",
        "calc_domain_max",
        "calc_domain_min",
        "speed_of_sound",
        "shifting",
        "acceleration_limiter",
        "sigma",
        "dynamic_time_step",
        "surface_tension",
        "gravity_csv",
        "rotation_axis",  # previously rotational_axis
        "rotation_center",  # previously rotational_support
        "rotating_system",  # New; False by default; required for rotating system
        "rotation_velocity",  # previously angulat_velocity
        "thermodynamics",
        "ambient_temperature",
    ]:
        if key in options:
            settings[key] = options[key]

    simulation.update_settings(Settings(**settings))
    # TODO: create default values if a key is missing in fillings
    for filling in fillings:
        if "speed of sound" not in fillings and "speed_of_sound" in settings:
            fillings[filling]["speed of sound"] = settings["speed_of_sound"]
        if (
            "isentropic exponent" in fillings[filling]
            and type(fillings[filling]["isentropic exponent"]) == str
        ):
            if fillings[filling]["isentropic exponent"] == "liquid":
                fillings[filling]["isentropic exponent"] = 7.0
            elif fillings[filling]["isentropic exponent"] == "gas":
                fillings[filling]["isentropic exponent"] = 1.4

    ####
    # provide a slip wall material
    # slip_def = BoundaryMaterial(name="slip", slip_wall=True)
    # slip = simulation.use_material(slip_def)
    # generate boundary materials
    materials = {}
    for name, part in parts.items():
        mtrl = ""
        inputs = {}
        if "slip" in part:  #
            # inputs["slip_wall"] = part["slip"]
            if part["slip"] in [True, "slip"]:
                inputs["velocity_boundary_condition"] = "slip"
                mtrl += f"slip_"
        if "contact_angle" in part:
            inputs["contact_angle"] = part[
                "contact_angle"
            ]  # TODO:check whether surface tension  is on
            mtrl += f"angle:{inputs['contact_angle'] }_"
        elif "contact_angle" in options:
            inputs["contact_angle"] = options[
                "contact_angle"
            ]  # use global contact_amgle of provided
            mtrl += f"angle:{inputs['contact_angle'] }_"

        if "virtual_movement" in part:  #
            if part["virtual_movement"] in [True, "moving"]:
                inputs["velocity_boundary_condition"] = "moving"
                inputs_vel = insert2rotatory(part, options)
                inputs_vel.pop("start", None)
                inputs_vel.pop("end", None)
                inputs["moving_boundary_condition"] = RotatoryMotion(
                    name=name,
                    **inputs_vel,
                )
                mtrl += f"moving_{part['omega']}_"
                part.pop("omega", None)  # prevent normal kinematics loop
        if "velocity_boundary_condition" in part:
            inputs["velocity_boundary_condition"] = part[
                "velocity_boundary_condition"
            ]  # "slip", "moving", "no-slip"
            mtrl += (
                f"velocity_boundary_condition_{inputs['velocity_boundary_condition']}"
            )
            if part["velocity_boundary_condition"] == "moving":
                inputs_vel = insert2rotatory(part, options)
                inputs_vel.pop("start", None)
                inputs_vel.pop("end", None)
                inputs["moving_boundary_condition"] = RotatoryMotion(
                    name=name,
                    **inputs_vel,
                )
                mtrl += f"{part['omega']}_"
                part.pop("omega", None)  # prevent normal kinematics loop
        if (
            mtrl != ""
        ):  # slip_wall or not contact_angle== None or not temperature==None or not adiabatic:
            if "material" in part:
                # TODO: check whether other stuff is set
                material = part["material"]
            else:
                material = mtrl  # f"angle:{contact_angle}_slip:{slip_wall}"
                part["material"] = material
            if material not in materials:
                materials[material] = inputs

    def set_fillings(fillings, unit="m"):
        for name, filling in fillings.items():
            input_viscosity = {"viscosity": filling["viscosity"]}
            if "type" in filling:
                if filling["type"].lower() in [
                    "non-newtonian",
                    "non_newtonian",
                    "non newtonian",
                    "shear rate",
                ]:
                    input_viscosity["viscosity_kind"] = (
                        MaterialViscosityKind.NON_NEWTONIAN
                    )
                elif filling["type"].lower() in [
                    "temperature",
                    "temperature_dependent",
                    "temperature-dependent",
                    "temperature dependent",
                ]:
                    input_viscosity["viscosity_kind"] = (
                        MaterialViscosityKind.TEMPERATURE_DEPENDENT
                    )
                else:
                    f"Filling type {filling['type']} not defined. Use 'non-newtonian' or 'temperature' instead."
                if type(filling["viscosity"]) in [str, dict]:
                    input_viscosity["viscosity"] = Datatable(filling["viscosity"])
                else:
                    f"Filling data source {filling['viscosity']} must be provided as path (str) or dict with keys 'shear_rate' or 'temperature' with 'viscosity'."
            if "specific_heat_capacity" and "thermal_conductivity" in filling:
                try:
                    phase_material_id = simulation.use_material(
                        FluidMaterial(
                            name=filling["material"],
                            **input_viscosity,
                            density=filling["density"],
                            speed_of_sound=filling["speed of sound"],
                            isentropic_exp=filling["isentropic exponent"],
                            specific_heat_capacity=filling["specific_heat_capacity"],
                            thermal_conductivity=filling["thermal_conductivity"],
                        )
                    )
                except:
                    print(
                        "You need to provide the specific heat capacity and thermal conductivity as a material."
                    )
            elif "density" and "isentropic exponent" in filling:
                phase_material_id = simulation.use_material(
                    FluidMaterial(
                        name=filling["material"],
                        **input_viscosity,
                        density=filling["density"],
                        speed_of_sound=filling["speed of sound"],
                        isentropic_exp=filling["isentropic exponent"],
                    )
                )
        for name, filling in fillings.items():
            phase_material = []

            # seed point check
            if "seed point" not in filling:
                filling["seed point"] = [None, None, None]

            if "fluid" in filling:
                try:
                    phase_material = simulation.get_materials(name=filling["fluid"])
                    phase_material_id = phase_material[0].id
                except:
                    phase_material = []
            if "material" in filling:
                try:
                    phase_material = simulation.get_materials(name=filling["material"])
                    phase_material_id = phase_material[0].id
                except:
                    phase_material = []
            if ("seed point" in filling) and ("level" in filling):
                filling["dive-object"] = simulation.add_filling(
                    name=name,
                    seed_point=filling["seed point"],
                    material=phase_material_id,
                )
                filling["dive-object"].set_height(filling["level"])
                if "temperature" in filling:
                    temperature = Component(
                        type=ComponentType.FILLING,
                        initial_temperature=filling["temperature"],
                    )
                    filling["dive-object"].update(component=temperature)
                if "initial_temperature" in filling:
                    temperature = Component(
                        type=ComponentType.FILLING,
                        initial_temperature=filling["initial_temperature"],
                    )
                    filling["dive-object"].update(component=temperature)
                if "initial_pressure" in filling:
                    initial_pressure = Component(
                        type=ComponentType.FILLING,
                        initial_pressure=filling["initial_pressure"],
                    )
                    filling["dive-object"].update(component=initial_pressure)
                if "initial_velocity" in filling:
                    initial_velocity = Component(
                        type=ComponentType.FILLING,
                        initial_velocity=filling["initial_velocity"],
                    )
                    filling["dive-object"].update(component=initial_velocity)
            elif ("seed point" in filling) and ("stl" in filling):
                # stl upload; enable unit conversion
                if unit in ["meter", "m", "meters", "metre", "metres"]:
                    unit = Unit.METER
                elif unit in ["inch", "in", "inches"]:
                    unit = Unit.INCHES
                elif unit in [
                    "millimeter",
                    "mm",
                    "millimeters",
                    "millimetre",
                    "millimetres",
                ]:
                    unit = Unit.MILLIMETER
                else:
                    print(
                        f'INFO: Unit {unit}  not defined. Available units are "m", "mm", and "in". Using "m" as default.'
                    )
                    unit = Unit.MILLIMETER

                # check whether unit is specified and set it. Use meters as default
                if "unit" in filling:
                    if filling["unit"] in [
                        "meter",
                        "m",
                        "meters",
                        "metre",
                        "metres",
                    ]:
                        filling_unit = Unit.METER
                    elif filling["unit"] in ["inch", "in", "inches"]:
                        filling_unit = Unit.INCHES
                    elif filling["unit"] in [
                        "millimeter",
                        "mm",
                        "millimeters",
                        "millimetre",
                        "millimetres",
                    ]:
                        filling_unit = Unit.MILLIMETER
                    else:
                        print(
                            f'INFO: Unit {filling["unit"]} of filling {name} not defined. Available units are "m", "mm", and "in". Using {unit} as default.'
                        )
                else:
                    filling_unit = unit
                # stl upload
                filling["dive-object"] = simulation.add_fluid(
                    name,
                    filling["seed point"],
                    phase_material_id,
                    filepath=filling["stl"],
                    shape_type=ShapeType.STL,
                    unit=filling_unit,
                )
                if "temperature" in filling:
                    temperature = Component(
                        type=ComponentType.FLUID,
                        initial_temperature=filling["temperature"],
                    )
                    filling["dive-object"].update(component=temperature)
                if "initial_temperature" in filling:
                    temperature = Component(
                        type=ComponentType.FLUID,
                        initial_temperature=filling["initial_temperature"],
                    )
                    filling["dive-object"].update(component=temperature)
                if "initial_pressure" in filling:
                    initial_pressure = Component(
                        type=ComponentType.FLUID,
                        initial_pressure=filling["initial_pressure"],
                    )
                    filling["dive-object"].update(component=initial_pressure)
                if "initial_velocity" in filling:
                    initial_velocity = Component(
                        type=ComponentType.FLUID,
                        initial_velocity=filling["initial_velocity"],
                    )
                    filling["dive-object"].update(component=initial_velocity)

    def add_refinement_zone(refinement_zones: dict):
        # Initialize a mesh that particles will be refined in
        for name, refinement_zone in refinement_zones.items():
            if "radius" in refinement_zone:
                if "axis" not in refinement_zone:
                    refinement_zone["axis"] = [0, 0, 1]
                if "center" not in refinement_zone:
                    refinement_zone["center"] = [0, 0, 0]
                refinement_zone["zone"] = Cylinder(
                    radius=refinement_zone["radius"],  # meter
                    height=refinement_zone["height"],  # meter
                    axis=refinement_zone["axis"],  # rotation axis
                    translation=refinement_zone["center"],  # move center
                )

            else:
                if "dimension" not in refinement_zone:
                    refinement_zone["dimension"] = [1, 1, 1]
                if "center" not in refinement_zone:
                    refinement_zone["center"] = [0, 0, 0]
                if "rotation" not in refinement_zone:
                    refinement_zone["rotation"] = [0, 0, 0]

                refinement_zone["zone"] = Cube(
                    dimensions=refinement_zone["dimension"],  # meter
                    rotation=refinement_zone["rotation"],  # degrees
                    translation=refinement_zone["center"],  # move center
                )
            # Cylinders can also be used
            # Add to simulation
            print(f"Refinement zones are added.")
            refinement = simulation.add_refinement_zone(
                name=name, mesh=refinement_zone["zone"]
            )

    if "rainbow" in options and options["rainbow"] == True:
        color_parts(parts)

    def add_sensors(sensors: dict):
        for name, sensor in sensors.items():
            shape_type = sensor.get("shape_type", None)

            if shape_type:
                shape_type = shape_type.lower()

            if shape_type == "point":
                add_point_sensor(name, sensor)
            else:
                add_surface_sensor(name, sensor, shape_type)

    def add_point_sensor(name, sensor):
        if "parameters" not in sensor:
            raise ValueError(
                f"No position defined for point sensor '{name}'. ('position':[x,y,z])"
            )
        simulation.add_point_sensor(name=name, **sensor["parameters"])

    def add_surface_sensor(name, sensor, shape_type):
        if shape_type is None:
            if not os.path.exists(sensor["stl"]):
                raise ValueError(
                    f"Path {sensor['stl']} is not valid for sensor '{name}'."
                )
            if sensor.get("parameters") is None:
                mesh = ComplexMesh(filepath=sensor["stl"])
            else:
                mesh = ComplexMesh(filepath=sensor["stl"], **sensor["parameters"])
            simulation.add_surface_sensor(name=name, mesh=mesh)
        else:
            mesh_shape = create_mesh_shape(sensor, shape_type)
            simulation.add_surface_sensor(name=name, mesh=mesh_shape)

    def create_mesh_shape(sensor, shape_type):
        shape_classes = {
            # "cube": Cube,
            # "cylinder": Cylinder,
            "rectangle": Rectangle,
            "circle": Circle,
        }

        if shape_type in shape_classes:
            return shape_classes[shape_type](**sensor["parameters"])
        else:
            raise ValueError(f"Unknown shape type '{shape_type}'")

    """   
    #sensor definitions
    sensors = {
        "sensor_1":{
            "shape_type": "point",
            "parameters":{
                "position":[1,0,0]
                }
            },
        "sensor_2":{
            "shape_type": "rectangle",
            "parameters":{
                "width":2,
                "height":1,
                "normal":[1,0,0],
                "translation":[0,2,0]
                }
            }
        }
    """

    # auxiliary function for setting walls and inlets and their respective motion.
    # it takes a dictiodinary as set in the header section and correctly
    # instantiates these  as "parts".
    def set_parts(parts, unit="m"):
        if unit in ["meter", "m", "meters", "metre", "metres"]:
            unit = Unit.METER
        elif unit in ["inch", "in", "inches"]:
            unit = Unit.INCHES
        elif unit in ["millimeter", "mm", "millimeters", "millimetre", "millimetres"]:
            unit = Unit.MILLIMETER
        else:
            print(
                f'INFO: Unit {unit}  not defined. Available units are "m", "mm", and "in". Using "m" as default.'
            )
            unit = Unit.MILLIMETER
        # nested follower collection
        follower_list = []

        # part loop
        for name, part in parts.items():
            part["movement_ids"] = []
            # check whether unit is specified and set it. Use meters as default
            unit_scaling = 1
            if "unit" in part:
                if part["unit"] in ["meter", "m", "meters", "metre", "metres"]:
                    part_unit = Unit.METER
                elif part["unit"] in ["inch", "in", "inches"]:
                    part_unit = Unit.INCHES
                    unit_scaling = 0.0254
                elif part["unit"] in [
                    "millimeter",
                    "mm",
                    "millimeters",
                    "millimetre",
                    "millimetres",
                ]:
                    part_unit = Unit.MILLIMETER
                    unit_scaling = 0.001
                else:
                    print(
                        f'INFO: Unit {part["unit"]} of part {name} not defined. Available units are "m", "mm", and "in". Using {unit} as default.'
                    )
            else:
                part_unit = unit

            # check whether flip normals is set
            if "flip" in part:
                flip = part["flip"]
            else:
                flip = False

            # open boundary configurations
            if (
                "inlet_velocity" in part
                or "outlet_velocity" in part
                or "outlet_pressure" in part
                or "inlet_pressure" in part
                or "opening" in part
            ):
                if "fluid" in part:
                    phase_material = simulation.get_materials(name=part["fluid"])
                    phase_material_id = phase_material[0].id
                else:
                    phase_material_id = simulation.get_materials(name="Water 20°C")[
                        0
                    ].id

                if "csv" not in part:
                    if "inlet_velocity" in part:
                        ob = OpenBoundary(
                            OpenBoundaryDirection.INLET,
                            OpenBoundaryType.RIEMANN_VELOCITY,
                            velocity=part["inlet_velocity"],
                            changeDirection=flip,
                        )
                    if "outlet_velocity" in part:
                        ob = OpenBoundary(
                            OpenBoundaryDirection.OUTLET,
                            OpenBoundaryType.RIEMANN_VELOCITY,
                            velocity=part["outlet_velocity"],
                            changeDirection=flip,
                        )
                    if "outlet_pressure" in part:
                        ob = OpenBoundary(
                            OpenBoundaryDirection.OUTLET,
                            OpenBoundaryType.RIEMANN_PRESSURE,
                            pressure=part["outlet_pressure"],
                            changeDirection=flip,
                        )
                    if "inlet_pressure" in part:
                        ob = OpenBoundary(
                            OpenBoundaryDirection.INLET,
                            OpenBoundaryType.RIEMANN_PRESSURE,
                            pressure=part["inlet_pressure"],
                            changeDirection=flip,
                        )
                    if "opening" in part:
                        ob = OpenBoundary(
                            OpenBoundaryDirection.INLET,
                            OpenBoundaryType.RIEMANN_PRESSURE,
                            pressure=part["opening"],
                            changeDirection=flip,
                        )
                else:
                    if "inlet_velocity" in part:
                        ob = OpenBoundary(
                            OpenBoundaryDirection.INLET,
                            OpenBoundaryType.RIEMANN_VELOCITY,
                            changeDirection=flip,
                            velocity_csv=part["csv"],
                        )
                    if "outlet_velocity" in part:
                        ob = OpenBoundary(
                            OpenBoundaryDirection.OUTLET,
                            OpenBoundaryType.RIEMANN_VELOCITY,
                            changeDirection=flip,
                            velocity_csv=part["csv"],
                        )

                if "temperature" in part:
                    part["dive-object"] = simulation.add_open_boundary(
                        name,
                        part["stl"],
                        ob,
                        phase_material_id,
                        unit=part_unit,
                        initial_temperature=part["temperature"],
                    )
                elif "initial_temperature" in part:
                    part["dive-object"] = simulation.add_open_boundary(
                        name,
                        part["stl"],
                        ob,
                        phase_material_id,
                        unit=part_unit,
                        initial_temperature=part["initial_temperature"],
                    )
                else:
                    part["dive-object"] = simulation.add_open_boundary(
                        name, part["stl"], ob, phase_material_id, unit=part_unit
                    )
            elif "type" in part:
                if part["type"] == "RotationPeriodicBoundary":
                    if any(
                        s in part
                        for s in [
                            "angle",
                            "rotation_axis",
                            "rotation_center",
                            "edge_begin",
                            "edge_end",
                        ]
                    ):
                        print(
                            "Definition of periodic boundary changed. You have to use the new notation."
                        )
                    periodic_boundary = RotationPeriodicBoundary(
                        slice_angle=part["slice_angle"],
                        slice_rotation=part["slice_rotation"],
                        axis=part["axis"],
                        center=part["center"],
                    )
                    part["dive-periodic_boundary"] = simulation.set_periodic_boundary(
                        periodic_boundary
                    )

                if part["type"] == "TranslationPeriodicBoundary":
                    print("Translatoric periodic bc are not yet supported")
                    # periodic_boundary = TranslationPeriodicBoundary(
                    #     first_edge_begin=part["first_edge_begin"],
                    #     first_edge_end=part["first_edge_end"],
                    #     last_edge_begin=part["last_edge_begin"],
                    #     last_edge_end=part["last_edge_end"],
                    # )
                    # part["dive-periodic_boundary"] = simulation.add_periodic_boundary(
                    #     periodic_boundary
                    # )
            else:  # its a wall
                if "material" in part:
                    material_def = materials[
                        part["material"]
                    ]  # TODO: more powerful material definitions
                else:
                    material_def = {}
                tbc = None
                if "adiabatic" or "heat_transfer_coefficient" or "temperature" in part:
                    if "heat_transfer_coefficient" in part:
                        tbc = ThermoBoundaryCondition.set_heat_transfer_coefficient(
                            heat_transfer_coefficient=part["heat_transfer_coefficient"],
                            reference_temperature=part["reference_temperature"],
                        )
                    elif "heat_flux" in part:
                        tbc = ThermoBoundaryCondition.set_heat_flux(
                            heat_flux=part["heat_flux"]
                        )
                    elif "temperature" in part:
                        tbc = ThermoBoundaryCondition.set_temperature(
                            boundary_temperature=part["temperature"]
                        )
                    elif "adiabtic" in part:
                        if part["adiabatic"] == True:
                            tbc = ThermoBoundaryCondition.set_adiabatic()
                    part["dive-object"] = simulation.add_wall_boundary(
                        name,
                        part["stl"],
                        unit=part_unit,
                        flip_normals=flip,
                        thermo_boundary_condition=tbc,
                        **material_def,
                    )
                else:
                    part["dive-object"] = simulation.add_wall_boundary(
                        name,
                        part["stl"],
                        unit=part_unit,
                        flip_normals=flip,
                        **material_def,
                    )

            if ("color" in part) and ("type" not in part):
                part["dive-color"] = part["dive-object"].set_color(part["color"])
            if "wireframe" in part:
                part["dive-color"] = part["dive-object"].set_wireframe(
                    part["wireframe"]
                )
            # Repositionings
            # scale
            if "scale" in part:
                part["dive-scale"] = part["dive-object"].scale(part["scale"])

            # translate
            if "translate" in part:
                part["dive-translation"] = part["dive-object"].translate(
                    part["translate"]
                )
            # Rotate
            if "rotate" in part:
                part["dive-rotation"] = part["dive-object"].rotate(part["rotate"])

            # define translatory movements
            if "acc" in part:
                if type(part["acc"]) == str:
                    # mvmt_id = part["dive-object"].add_translatory_movement(
                    #     name=f"{name}_translatoric_acc",
                    #     acceleration_csv=part["acc"],
                    #     direction=part["axis"],
                    #     start=part["start"],
                    #     end=part["end"],
                    #     virtual_movement=part["virtual_movement"],
                    # )
                    mvmt_id = part["dive-object"].add_motion(
                        TranslatoryMotion(
                            name=f"{name}_translatoric_acc",
                            acceleration_csv=part["acc"],
                            direction=part["axis"],
                            start=part["start"],
                            end=part["end"],
                            # virtual_movement=part["virtual_movement"],
                        )
                    )
                    part["movement_ids"].append(mvmt_id)
                else:
                    # mvmt_id = part["dive-object"].add_translatory_movement(
                    #     name=f"{name}_translatoric_acc",
                    #     acc=part["acc"],
                    #     direction=part["axis"],
                    #     start=part["start"],
                    #     end=part["end"],
                    #     virtual_movement=part["virtual_movement"],
                    # )
                    mvmt_id = part["dive-object"].add_motion(
                        TranslatoryMotion(
                            name=f"{name}_translatoric_acc",
                            acc=part["acc"],
                            direction=part["axis"],
                            start=part["start"],
                            end=part["end"],
                            # virtual_movement=part["virtual_movement"],
                        )
                    )
                    part["movement_ids"].append(mvmt_id)
            if "vel" in part:
                # mvmt_id = part["dive-object"].add_translatory_movement(
                #     name=f"{name}_translatoric",
                #     vel=part["vel"],
                #     direction=part["axis"],
                #     start=part["start"],
                #     end=part["end"],
                # )
                mvmt_id = part["dive-object"].add_motion(
                    TranslatoryMotion(
                        name=f"{name}_translatoric",
                        vel=part["vel"],
                        direction=part["axis"],
                        start=part["start"],
                        end=part["end"],
                    )
                )
                part["movement_ids"].append(mvmt_id)

            # define rotatory movements
            # TODO: Consider moving this line
            part["dive-movement-leaders"] = []
            if options["start_time"] > 0:
                if "omega_leader" in part:
                    inputs = insert2rotatory(part, options, kinematic_type="leader")
                    if "start_motion" in part:
                        inputs["start"] = part["start_motion"]
                    else:
                        inputs["start"] = 0
                    inputs["end"] = options["start_time"]
                    del inputs["vel"]
                    inputs["acc"] = part["omega_leader"] / (
                        options["start_time"] - inputs["start"]
                    )

                    # mvmt_id = part["dive-object"].add_rotatory_movement(
                    #     name=f"{name}_leader_acc",
                    #     **inputs,
                    # )
                    mvmt_id = part["dive-object"].add_motion(
                        RotatoryMotion(
                            name=f"{name}_leader_acc",
                            **inputs,
                        )
                    )

                    part["dive-movement-leaders"].append(mvmt_id)
                    part["movement_ids"].append(mvmt_id)
                if (
                    "omega_follower" in part
                ):  # TODO: It seems like there is no real use case for follower and wherever it is used the standard case should also do the job...
                    inputs = insert2rotatory(part, options, kinematic_type="follower")
                    if "start_motion" in part:
                        inputs["start"] = part["start_motion"]
                    else:
                        inputs["start"] = 0
                    inputs["end"] = options["start_time"]
                    del inputs["vel"]
                    inputs["acc"] = part["omega_follower"] / (
                        options["start_time"] - inputs["start"]
                    )
                    # part["dive-movement-follower"] = part[
                    #     "dive-object"
                    # ].add_rotatory_movement(
                    #     name=f"{name}_follower_acc",
                    #     **inputs,
                    # )
                    part["dive-movement-follower"] = part["dive-object"].add_motion(
                        RotatoryMotion(
                            name=f"{name}_follower_acc",
                            **inputs,
                        )
                    )
                    part["movement_ids"].append(part["dive-movement-follower"])
                if "omega" in part:
                    inputs = insert2rotatory(part, options)
                    if "start_motion" in part:
                        inputs["start"] = part["start_motion"]
                    else:
                        inputs["start"] = 0
                    inputs["end"] = options["start_time"]
                    del inputs["vel"]
                    inputs["acc"] = part["omega"] / (
                        options["start_time"] - inputs["start"]
                    )
                    # part["dive-movement"] = part["dive-object"].add_rotatory_movement(
                    #     name=f"{name}_acc",
                    #     **inputs,
                    # )
                    part["dive-movement"] = part["dive-object"].add_motion(
                        RotatoryMotion(
                            name=f"{name}_acc",
                            **inputs,
                        )
                    )
                    part["movement_ids"].append(part["dive-movement"])

            if "omega_leader" in part:
                inputs = insert2rotatory(part, options, kinematic_type="leader")

                # mvmt_id = part["dive-object"].add_rotatory_movement(
                #     name=f"{name}_leader", **inputs
                # )
                mvmt_id = part["dive-object"].add_motion(
                    RotatoryMotion(
                        name=f"{name}_leader",
                        **inputs,
                    )
                )
                part["dive-movement-leaders"].append(mvmt_id)
                part["movement_ids"].append(mvmt_id)
            if "omega_follower" in part:
                inputs = insert2rotatory(part, options, kinematic_type="follower")
                # part["dive-movement-follower"] = part[
                #     "dive-object"
                # ].add_rotatory_movement(name=f"{name}_follower", **inputs)
                part["dive-movement-follower"] = part["dive-object"].add_motion(
                    RotatoryMotion(
                        name=f"{name}_follower",
                        **inputs,
                    )
                )
                part["movement_ids"].append(part["dive-movement-follower"])
            if "omega" in part:
                inputs = insert2rotatory(part, options)
                # part["dive-movement"] = part["dive-object"].add_rotatory_movement(
                #     name=name, **inputs
                # )
                part["dive-movement"] = part["dive-object"].add_motion(
                    RotatoryMotion(
                        name=name,
                        **inputs,
                    )
                )
                part["movement_ids"].append(part["dive-movement"])
            if "oscillation" in part:
                sig = +1
                for idx, val in enumerate(
                    np.r_[0 : options["end_time"] : part["delta_t"]]
                ):
                    # part[f"dive-movement{idx}"] = part[
                    #     "dive-object"
                    # ].add_rotatory_movement(
                    #     name=name + f"_{idx}",
                    #     start=val,
                    #     end=val + part["delta_t"],
                    #     vel=sig * part["oscillation"],
                    #     axis=part["axis"],
                    #     center=part["center"],
                    # )
                    part[f"dive-movement{idx}"] = part["dive-object"].add_motion(
                        RotatoryMotion(
                            name=name + f"_{idx}",
                            start=val,
                            end=val + part["delta_t"],
                            vel=sig * part["oscillation"],
                            axis=part["axis"],
                            center=part["center"],
                        )
                    )
                    sig *= -1
            if "oscillation_follower" in part:
                sig = +1

                if "oscillation_shift" in part:
                    shift = part["oscillation_shift"] * part["delta_t"]
                    # part[f"dive-movementshift"] = part[
                    #     "dive-object"
                    # ].add_rotatory_movement(
                    #     name=name + f"_shift",
                    #     start=0,
                    #     end=0 + shift,
                    #     vel=sig * part["oscillation_follower"],
                    #     axis=part["axis_follower"],
                    #     center=part["center_follower"],
                    #     leader_id=part["dive-movement-leaders"],
                    # )
                    part[f"dive-movementshift"] = part["dive-object"].add_motion(
                        RotatoryMotion(
                            name=name + f"_shift",
                            start=0,
                            end=0 + shift,
                            vel=sig * part["oscillation_follower"],
                            axis=part["axis_follower"],
                            center=part["center_follower"],
                            leader_id=part["dive-movement-leaders"],
                        )
                    )
                    sig *= -1
                else:
                    shift = 0

                for idx, val in enumerate(
                    np.r_[shift : options["end_time"] : part["delta_t"]]
                ):
                    # part[f"dive-movement{idx}"] = part[
                    #     "dive-object"
                    # ].add_rotatory_movement(
                    #     name=name + f"_{idx}",
                    #     start=val,
                    #     end=val + part["delta_t"],
                    #     vel=sig * part["oscillation_follower"],
                    #     axis=part["axis_follower"],
                    #     center=part["center_follower"],
                    #     leader_id=part["dive-movement-leaders"],
                    # )
                    part[f"dive-movement{idx}"] = part["dive-object"].add_motion(
                        RotatoryMotion(
                            name=name + f"_{idx}",
                            start=val,
                            end=val + part["delta_t"],
                            vel=sig * part["oscillation_follower"],
                            axis=part["axis_follower"],
                            center=part["center_follower"],
                            leader_id=part["dive-movement-leaders"],
                        )
                    )
                    sig *= -1
            if "oscillation_translation" in part:
                sig = +1
                for idx, val in enumerate(
                    np.r_[0 : options["end_time"] : part["delta_t"]]
                ):
                    # part[f"dive-movement{idx}"] = part[
                    #     "dive-object"
                    # ].add_translatory_movement(
                    #     name=name + f"_{idx}",
                    #     start=val,
                    #     end=val + part["delta_t"],
                    #     vel=sig * part["oscillation_translation"],
                    #     direction=part["axis_translation"],
                    # )
                    part[f"dive-movement{idx}"] = part["dive-object"].add_motion(
                        TranslatoryMotion(
                            name=name + f"_{idx}",
                            start=val,
                            end=val + part["delta_t"],
                            vel=sig * part["oscillation_translation"],
                            direction=part["axis_translation"],
                        )
                    )
                    sig *= -1
            if "automated_refinement" in part:
                print(f"Automated refinement will be done for {name}.")
                if part["automated_refinement"] == True and all(
                    k not in part
                    for k in (
                        "inlet_velocity",
                        "outlet_velocity",
                        "inlet_pressure",
                        "outlet_pressure",
                        "type",
                    )
                ):
                    refinement_zones[f"{name}_zone"] = automated_refinement(
                        part=part,
                        dr=settings["particle_diameter"],
                        unit_scaling=unit_scaling,
                    )
            if "follow" in part:
                follower_list.append(name)
        # follower loop
        for follower in follower_list:
            leader_movements = []
            for leader_name in parts[follower]["follow"]:
                part_fol = parts[follower]
                part_lead = parts[leader_name]
                part_lead["dive-movement-leaders"] = []
                if options["start_time"] > 0:
                    if "omega_leader" in part_lead:
                        inputs = insert2rotatory(
                            part_lead, options, kinematic_type="leader"
                        )
                        if "start_motion" in part_lead:
                            inputs["start"] = part_lead["start_motion"]
                        else:
                            inputs["start"] = 0
                        inputs["end"] = options["start_time"]
                        del inputs["vel"]
                        inputs["acc"] = part_lead["omega_leader"] / (
                            options["start_time"] - inputs["start"]
                        )

                        # mvmt_id = part_fol["dive-object"].add_rotatory_movement(
                        #     name=f"{leader_name}_leader_acc",
                        #     **inputs,
                        # )
                        mvmt_id = part_fol["dive-object"].add_motion(
                            RotatoryMotion(
                                name=f"{leader_name}_leader_acc",
                                **inputs,
                            )
                        )
                        part_lead["dive-movement-leaders"].append(mvmt_id)
                        leader_movements.append(mvmt_id)
                    if (
                        "omega_follower" in part_lead
                    ):  # TODO: It seems like there is no real use case for follower and wherever it is used the standard case should also do the job...
                        inputs = insert2rotatory(
                            part_lead, options, kinematic_type="follower"
                        )
                        if "start_motion" in part_lead:
                            inputs["start"] = part_lead["start_motion"]
                        else:
                            inputs["start"] = 0
                        inputs["end"] = options["start_time"]
                        del inputs["vel"]
                        inputs["acc"] = part_lead["omega_follower"] / (
                            options["start_time"] - inputs["start"]
                        )
                        # part_fol["dive-movement-follower"] = part_fol[
                        #     "dive-object"
                        # ].add_rotatory_movement(
                        #     name=f"{leader_name}_follower_acc",
                        #     **inputs,
                        # )
                        part_fol["dive-movement-follower"] = part_fol[
                            "dive-object"
                        ].add_motion(
                            RotatoryMotion(
                                name=f"{leader_name}_follower_acc",
                                **inputs,
                            )
                        )
                        leader_movements.append(part_fol["dive-movement-follower"])
                    if "omega" in part_lead:
                        inputs = insert2rotatory(part_lead, options)
                        if "start_motion" in part:
                            inputs["start"] = part_lead["start_motion"]
                        else:
                            inputs["start"] = 0
                        inputs["end"] = options["start_time"]
                        del inputs["vel"]
                        inputs["acc"] = part_lead["omega"] / (
                            options["start_time"] - inputs["start"]
                        )
                        # part_lead["dive-movement"] = part_fol[
                        #     "dive-object"
                        # ].add_rotatory_movement(
                        #     name=f"{leader_name}_acc",
                        #     **inputs,
                        # )
                        part_lead["dive-movement"] = part_fol["dive-object"].add_motion(
                            RotatoryMotion(
                                name=f"{leader_name}_acc",
                                **inputs,
                            )
                        )
                        leader_movements.append(part_lead["dive-movement"])

                if "omega_leader" in part_lead:
                    inputs = insert2rotatory(
                        part_lead, options, kinematic_type="leader"
                    )
                    # mvmt_id = part_fol["dive-object"].add_rotatory_movement(
                    #     name=f"{leader_name}_leader", **inputs
                    # )
                    mvmt_id = part_fol["dive-object"].add_motion(
                        RotatoryMotion(
                            name=f"{leader_name}_leader",
                            **inputs,
                        )
                    )
                    part_lead["dive-movement-leaders"].append(mvmt_id)
                    leader_movements.append(mvmt_id)
                if "omega_follower" in part_lead:
                    inputs = insert2rotatory(
                        part_fol, options, kinematic_type="follower"
                    )
                    # part_lead["dive-movement-follower"] = part_fol[
                    #     "dive-object"
                    # ].add_rotatory_movement(name=f"{leader_name}_follower", **inputs)
                    part_lead["dive-movement-follower"] = part_fol[
                        "dive-object"
                    ].add_motion(
                        RotatoryMotion(
                            name=f"{leader_name}_follower",
                            **inputs,
                        )
                    )
                    leader_movements.append(part_lead["dive-movement-follower"])
                if "omega" in part_lead:
                    inputs = insert2rotatory(part_lead, options)
                    # part_lead["dive-movement"] = part_fol[
                    #     "dive-object"
                    # ].add_rotatory_movement(name=leader_name, **inputs)
                    part_lead["dive-movement"] = part_fol["dive-object"].add_motion(
                        RotatoryMotion(
                            name=leader_name,
                            **inputs,
                        )
                    )
                    leader_movements.append(part_lead["dive-movement"])
            # movement_id_leaders = []
            # for leader_name in parts[follower]["follow"]:
            #     movement_id_leaders.extend(parts[leader_name]["movement_ids"])
            for follower_movement in parts[follower]["movement_ids"]:
                parts[follower]["dive-object"]._Part__link_leaders(
                    leader_movements, follower_movement
                )

    set_fillings(fillings)
    set_parts(parts)
    add_refinement_zone(refinement_zones)
    add_sensors(sensors=sensors)

    if "calc_domain_active" in options:
        simulation.update_settings(
            Settings(calc_domain_active=options["calc_domain_active"])
        )
    return simulation


def add_sensors_post(sensors={}):
    # accessing the project
    try:
        project = Project.get(name=sensors["project_name"])[0]
    except IndexError:
        raise ValueError("project_name is not valid")

    try:
        simulation = project.get_simulations(name=sensors["simulation_name"])[0]
    except IndexError:
        raise ValueError("simulation_name is not valid")

    del sensors["project_name"]
    del sensors["simulation_name"]

    # Sensor functionality
    def add_sensors(sensors: dict):
        for name, sensor in sensors.items():
            shape_type = sensor.get("shape_type", None)

            if shape_type:
                shape_type = shape_type.lower()

            if shape_type == "point":
                add_point_sensor(name, sensor)
            else:
                add_surface_sensor(name, sensor, shape_type)

    def add_point_sensor(name, sensor):
        if "parameters" not in sensor:
            raise ValueError(
                f"No position defined for point sensor '{name}'. ('position':[x,y,z])"
            )
        simulation.add_point_sensor(name=name, **sensor["parameters"], to_last_run=True)

    def add_surface_sensor(name, sensor, shape_type):
        if shape_type is None:
            if not os.path.exists(sensor["stl"]):
                raise ValueError(
                    f"Path {sensor['stl']} is not valid for sensor '{name}'."
                )
            if sensor.get("parameters") is None:
                mesh = ComplexMesh(filepath=sensor["stl"])
            else:
                mesh = ComplexMesh(filepath=sensor["stl"], **sensor["parameters"])
            simulation.add_surface_sensor(name=name, mesh=mesh, to_last_run=True)
        else:
            mesh_shape = create_mesh_shape(sensor, shape_type)
            simulation.add_surface_sensor(name=name, mesh=mesh_shape, to_last_run=True)

    def create_mesh_shape(sensor, shape_type):
        shape_classes = {
            # "cube": Cube,
            # "cylinder": Cylinder,
            "rectangle": Rectangle,
            "circle": Circle,
        }

        if shape_type in shape_classes:
            return shape_classes[shape_type](**sensor["parameters"])
        else:
            raise ValueError(f"Unknown shape type '{shape_type}'")

    add_sensors(sensors=sensors)
    return simulation


def eval(config={}):
    # units
    ylabels_integral_available = {
        "volume_flux": "Volume Flow Rate (m³/s)",
        "mass_flux": "Mass Flow Rate (kg/s)",
        "wetting": "Wetting (m²)",
        "power_density": "Power (W)",
        "mass": "Mass (kg)",
        "heat_flux": "Heat Flux (W)",
    }
    ylabels_nonintegral_available = {
        "volume_flux": "Volume Flux (m/s)",
        "mass_flux": "Mass Flux (kg/(m²s))",
        "wetting": "Wetting (-)",
        "power_density": "Power Density (W/m²)",
        "pressure": "Pressure (Pa)",
        "shear_rate": "Shear Rate (1/s)",
        "particle_packing": "Particel Packing (-)",
        "mass": "Mass (-)",
        "kinematic_viscosity": "Kinematic Viscosity (m²/s)",
        "density": "Density (kg/m³)",
        "acceleration": "Acceleration (m/s²)",
        "acceleration_magnitude": "Acceleration (m/s²)",
        "normal_stress": "Normal Stress (N/m²)",
        "normal_stress_magnitude": "Normal Stress (N/m²)",
        "shear_stress_magnitude": "Shear Stress (N/m²)",
        "shear_stress": "Shear Stress (N/m²)",
        "total_stress": "Total Stress (N/m²)",
        "total_stress_magnitude": "Total Stress (N/m²)",
        "velocity_magnitude": "Velocity Magnitude (m/s)",
        "velocity": "Velocity (m/s)",
        "heat_flux": "Heat Flux (W/m²)",
        "temperature": "Temperature (K)",
    }

    # accessing the project
    def replace_average_with_mean(d):
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = replace_average_with_mean(value)
            elif isinstance(value, str):
                d[key] = value.replace("average", "mean")
            elif isinstance(value, list):
                d[key] = [
                    replace_average_with_mean(item) if isinstance(item, dict) else item
                    for item in value
                ]
        return d

    def concatenate_strings(input_list):
        if isinstance(input_list, list):
            return "_".join(str(item) for item in input_list if isinstance(item, str))
        else:
            return input_list

    def get_suffix(column_name):
        value = column_name.rsplit(":", 1)
        return value[-1] if len(value) > 1 else None

    def get_prefix(column_name):
        value = column_name.rsplit(" - ", 1)
        return value[0] if len(value) > 1 else None

    def plot_df(df, name):

        patterns = ["-", "--", "-.", ":"]

        colors = [
            "#2D69B2",
            "#154687",
            "#4C4C4C",
            "#25C189",
            "#0B5B38",
            "#FFD766",
            "#D64265",
            "#214779",
            "#ffb518",
            "#16885c",
            "#2761a8",
            "#df5a51",
            "#912f5c",
            "#255da3",
            "#c93c60",
            "#ffd766",
            "#0d6641",
            "#cf3f62",
            "#2a65ad",
            "#ffb922",
            "#199365",
            "#2d486c",
            "#1d5195",
            "#0b5b38",
            "#e3c960",
            "#929f51",
            "#1c9f6e",
            "#1f559a",
            "#ffb10f",
            "#17498b",
            "#a42a54",
            "#2d69b2",
            "#b03058",
            "#843667",
            "#bd365c",
            "#6b447c",
            "#525392",
            "#394a5f",
            "#25c189",
            "#3961a7",
            "#aa2d56",
            "#1a4d90",
            "#9e2852",
            "#334966",
            "#ffd766",
            "#77924c",
            "#274873",
            "#10714a",
            "#137d53",
            "#e36748",
            "#454b52",
            "#5c8447",
            "#ffd25c",
            "#ffb10f",
            "#9e2852",
            "#faa418",
            "#da4e5b",
            "#3f4a59",
            "#b6335a",
            "#e8733e",
            "#f18c2b",
            "#26683d",
            "#417642",
            "#ffbd2c",
            "#22b580",
            "#ec7f35",
            "#c8bb5b",
            "#d64265",
            "#2d69b2",
            "#c3395e",
            "#4c4c4c",
            "#ffce52",
            "#1b4680",
            "#d64265",
            "#adad56",
            "#1faa77",
            "#783d72",
            "#22599e",
            "#465a9c",
            "#ffca49",
            "#5f4c87",
            "#0b5b38",
            "#ffc135",
            "#f59822",
            "#ffc63f",
        ]
        custom_dashes = [
            (i, j, k, l)
            for i in range(1, 5)
            for j in range(1, 5)
            for k in range(1, 5)
            for l in range(1, 5)
        ]
        custom_dashes = custom_dashes[:96]
        line_styles = patterns + custom_dashes

        width_px = 1920
        height_px = 1080
        dpi = 150
        width_in = width_px / dpi
        height_in = height_px / dpi
        suffix_groups = {}

        for column in df.columns:
            suffix = get_suffix(column)
            if suffix:
                if suffix not in suffix_groups:
                    suffix_groups[suffix] = []
                suffix_groups[suffix].append(column)

        for suffix, columns in suffix_groups.items():

            if len(columns) < 50:
                fig, axs = plt.subplots(1, figsize=(width_in, height_in))
                for i, column in enumerate(columns):
                    if isinstance(line_styles[i], tuple):
                        axs.plot(
                            df["Time"],
                            df[column],
                            label=column,
                            color=colors[i],
                            linestyle="-",
                            dashes=line_styles[i],
                            linewidth=2.5,
                        )
                    else:
                        axs.plot(
                            df["Time"],
                            df[column],
                            label=column,
                            color=colors[i],
                            linestyle=line_styles[i],
                            linewidth=2.5,
                        )
                directions = ["X", "Y", "Z"]
                if suffix[-1] in directions:
                    attr_temp = get_prefix(suffix)
                    attr = get_prefix(attr_temp)
                else:
                    attr = get_prefix(suffix)

                if "integral" in suffix:
                    ylabel = ylabels_integral_available[attr.replace(" ", "")]
                else:
                    ylabel = ylabels_nonintegral_available[attr.replace(" ", "")]

                axs.set(
                    xlabel="Time (s)",
                    ylabel=ylabel,
                    xlim=(df["Time"].min(), df["Time"].max()),
                )
                axs.grid(True)
                plt.legend(loc="upper left", bbox_to_anchor=(1, 1.015))
                # image_path = f'{name}_({suffix}).png'
                image_path = f"{name}_{suffix}.png"
                plt.savefig(image_path, bbox_inches="tight")
            else:
                pass

    def add_units(df):
        for column in df.columns[1:]:
            suffix = get_suffix(column)

            directions = ["X", "Y", "Z"]

            if suffix[-1] in directions:
                attr_temp = get_prefix(suffix)
                attr = get_prefix(attr_temp)
            else:
                attr = get_prefix(suffix)

            if "integral" in suffix:
                unit = (
                    ylabels_integral_available[attr.replace(" ", "")]
                    .rsplit(" ", -1)[-1]
                    .replace(" ", "")
                )
                df = df.rename(columns={column: column + " " + unit})
            else:
                unit = (
                    ylabels_nonintegral_available[attr.replace(" ", "")]
                    .rsplit(" ", -1)[-1]
                    .replace(" ", "")
                )
                df = df.rename(columns={column: column + " " + unit})
        return df

    try:
        project = Project.get(name=config["project_name"])[0]
    except IndexError:
        raise ValueError("project_name is not valid")

    try:
        simulation = project.get_simulations(name=config["simulation_name"])[0]
    except IndexError:
        raise ValueError("simulation_name is not valid")

    replace_average_with_mean(config)
    time_step = simulation.data["settings"]["output_time_interval"]
    noaggr = False
    # getting the results

    with simulation.results_session() as results:
        print(results.available_outputs)
        outputs = config.get("output_range") or results.available_outputs
        output = [num for num in outputs if num in results.available_outputs]
        df = pd.DataFrame({"Output": output})
        names = "all"
        obj_kinds = "all"
        if config.get("names") == "all_parts":
            obj_kinds = ["wall_boundary", "open_boundary"]
        elif config.get("names") == "all_sensors":
            obj_kinds = ["volume", "point", "surface"]
        elif config.get("names") == "all_materials":
            obj_kinds = ["material"]
        elif config.get("names") == "all":
            obj_kinds = "all"
        else:
            names = config.get("names")

        attributes = config.get("attributes")
        operations = config.get("operations")

        num_parts = len(results.info)
        count_parts = 1
        computed = 0
        index = []
        values = []

        # df["Time"]=np.arange(df.iloc[0,0]*time_step,df.iloc[-1,0]*time_step + inc ,time_step)
        df["Time"] = np.array(output) * time_step
        df.index = output

        for obj in results.info:
            if computed >= 80:
                print(
                    "Warning: Too many results are calculated. This can lead to a loss of performance."
                )
            print(f"Object: {obj.name} [{count_parts}/{num_parts}]")
            count_parts += 1
            if names != "all" and obj.name not in names:
                print(f"Not using {obj.name}")
                continue
            if obj_kinds != "all" and obj.kind not in obj_kinds:
                print(f"Not using {obj.name}")
                continue
            missing_operations = {k: {} for k in attributes}
            for attribute in obj.attributes:
                if attribute.name not in missing_operations:
                    continue
                missing_ops = set(operations) - set(attribute.operations)
                if missing_ops:
                    missing_operations[attribute.name] = missing_ops
                else:
                    del missing_operations[attribute.name]

                if isinstance(operations, dict):
                    available_ops = {
                        k: v for k, v in operations.items() if k in attribute.operations
                    }
                else:
                    available_ops = list(set(operations) & set(attribute.operations))

                if available_ops:
                    print("Calculating Statistics")
                    results_stats, results_aggr = results.calc(
                        attribute.name,
                        operations=available_ops,
                        names=obj.name,
                        outputs=output,
                    )
                else:
                    print("operations not available")
                if isinstance(operations, dict) and results_aggr != {}:
                    values_temp = [
                        results_aggr[operation].values
                        for operation in operations.keys()
                        if operation in results_aggr
                    ]
                    index_temp = [
                        f"{obj.name}: {attribute.name} - {operation}"
                        for operation in available_ops
                        if operation in results_aggr
                    ]
                    noaggr = False
                else:
                    noaggr = True

                df_temp = pd.DataFrame()

                for operation in available_ops:
                    df_op = results_stats[operation].dataframe
                    df_temp = pd.concat([df_op, df_temp], axis=1)

                df_temp = df_temp.add_prefix(f"{obj.name}: ")
                df = pd.concat([df, df_temp], axis=1)

                if isinstance(operations, dict) and results_aggr != {}:
                    values = values + values_temp
                    index = index + index_temp

            computed += 1

            if missing_operations:
                print(
                    f"Some attributes and/or operations are not available for object: {obj.name}"
                )
                print(missing_operations)
                noaggr = False

        if noaggr is False:
            df2 = pd.DataFrame(values)
            df2.index = index
            df2 = df2.fillna("")

    if noaggr is False:
        indx_map = {}
        for idx in df2.index:
            ops = idx.rsplit(" ", -1)[-1].replace(" ", "")
            suffix = get_suffix(idx)
            attr = get_prefix(suffix)
            if "integral" == ops:
                indx_map[idx] = (
                    idx
                    + " "
                    + ylabels_integral_available[attr.replace(" ", "")]
                    .rsplit(" ", -1)[-1]
                    .replace(" ", "")
                )
            else:
                indx_map[idx] = (
                    idx
                    + " "
                    + ylabels_nonintegral_available[attr.replace(" ", "")]
                    .rsplit(" ", -1)[-1]
                    .replace(" ", "")
                )
        df2.rename(index=indx_map, inplace=True)
    else:
        df2 = pd.DataFrame()

    df.drop("Output", axis=1, inplace=True)

    # Plotting and saving
    plot = config.get("plot", False)
    directory = config.get("filepath", None)

    if directory is not None:

        base_filename = concatenate_strings(config.get("names"))
        extension = ".csv"
        counter = 0
        if not os.path.exists(directory):
            os.makedirs(directory)

        filename = f"{base_filename}_stats_{counter}{extension}"
        filename_2 = f"{base_filename}_aggr_{counter}{extension}"
        file_path = os.path.join(directory, filename)
        file_path_2 = os.path.join(directory, filename_2)

        while os.path.exists(file_path):
            counter += 1
            filename = f"{base_filename}_stats_{counter}{extension}"
            filename_2 = f"{base_filename}_aggr_{counter}{extension}"
            file_path = os.path.join(directory, filename)
            file_path_2 = os.path.join(directory, filename_2)
        if plot:
            plot_df(df, file_path[:-4])
        df = add_units(df)
        df.to_csv(file_path, encoding="utf-8-sig")
        if noaggr is False:
            df2.to_csv(file_path_2, encoding="utf-8-sig")

    return df, df2

# %%

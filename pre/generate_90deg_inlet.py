import os
import numpy as np
from stl import mesh    # pip install numpy-stl
from scipy.spatial import ConvexHull
import pyvista as pv
#######################################################################################
# READ ME
"""
Generate flat spray inlet based on flowrate, spray angle and inlet size.
Uses the corrected cos/sin velocity formulation (constant magnitude, true
direction = alpha) instead of the original (v, v*tan(alpha)) which inflates
edge speeds.

Minimum particle size = plane height / 6 for single inlet.
Multi-inlet setting not recommended (spray is over-simplified).

==== Spray-angle calibration ====================================================
With the cos/sin formulation, the *commanded* spray angle is the true geometric
sweep of velocity directions across the inlet. However, the *measured* visible
spray angle tends to come out smaller than the commanded value because:
    - cos-flux falloff at the edges: only v*cos(alpha) is normal to the inlet,
      so fewer particles per unit time emerge at the edges.
    - air drag and finite particle-count edge sparsity make the visible spray
      "taper".

Two ways to compensate:
    (1) Overshoot the commanded angle (set `spray_angle` larger). Single line.
    (2) Re-introduce the (cos, tan) formulation -- boosts edge velocity to
        compensate cos-flux loss. Toggle EDGE_BOOST_FORMULATION = True below.

The default below uses option (1): a commanded angle slightly higher than the
target. Tweak `spray_angle_target_deg` and `calibration_factor` to taste.
=================================================================================
"""
#######################################################################################
# REQUIRED PACKAGES
# pip install numpy-stl scipy numpy pyvista
# pip install dive-fisherman --extra-index-url https://pkgs.dev.azure.com/divesph/dive-public/_packaging/dive-registry/pypi/simple/
#######################################################################################
path = os.path.dirname(os.path.abspath(__file__))+'/'   # path of the current script
#######################################################################################
project_name = 'Script Velocity inlet'
sim_name     = 'Test_calibrated'

inlet_plane_stl = None  # path to STL or None to auto-build a rectangle

# Used only if inlet_plane_stl is None
rectangle_plane = {
    'center': [0, 0, 0],
    'width':  3.5e-4,
    'height': 4e-3,
    'normal': [0, 1, 0],          # sign matters!
    'angle':  0,
}

# ---- Spray parameters ----------------------------------------------------------
spray_angle_target_deg = 90       # what you WANT to measure on the protractor [deg]
calibration_factor     = 90.0 / 80.5   # compensates the measured ~80.5 deg shortfall
                                         # -> commands ~100.6 deg, iterate as needed

d_p              = 0.0001         # m, target particle diameter
volume_flowrate  = 1.53           # L/min
upload_to_sim    = True
multi_inlet      = False          # False recommended

# Pick the velocity formulation:
#   False -> v = (v_in*cos(a), v_in*sin(a))  (constant magnitude, true direction)
#   True  -> v = (v_in,        v_in*tan(a))  (constant normal flux, edges faster)
EDGE_BOOST_FORMULATION = False

#######################################################################################
# corrections automatically used for multi inlets
flowrate_correction   = True
sub_inlet_correction  = True
#######################################################################################
def save_3D_field(points, vector_data=None, scalar_data=None,
                  vector_names=[], scalar_names=[], file_name=''):
    points = np.asarray(points, dtype=float)

    if vector_data is not None:
        vector_data = np.asarray(vector_data, dtype=float)
        n_vectors = vector_data.shape[2] if len(vector_data.shape) > 2 else 1

    if scalar_data is not None:
        scalar_data = np.asarray(scalar_data, dtype=float)
        n_scalar = scalar_data.shape[1] if len(scalar_data.shape) > 1 else 1

    m = pv.PolyData(points)

    if vector_data is not None:
        for i in range(n_vectors):
            if n_vectors == 1:
                m.point_data[vector_names[i]] = vector_data[:, :]
            else:
                m.point_data[vector_names[i]] = vector_data[:, i, :]

    if scalar_data is not None:
        for i in range(n_scalar):
            if n_scalar == 1:
                m.point_data[scalar_names[i]] = scalar_data[:]
            else:
                m.point_data[scalar_names[i]] = scalar_data[:, i]

    m.save(file_name)


def rectangle_from_center_width_height(center, width, height, normal, angle, filename):
    C = np.asarray(center, float)
    N = np.asarray(normal, float)

    n_norm = np.linalg.norm(N)
    if n_norm == 0:
        raise ValueError("Normal vector must be non-zero")
    n = N / n_norm

    a = width  / 2.0
    b = height / 2.0
    e1, e2 = surface_tangents(n)

    if angle != 0.0:
        cos_a = np.cos(angle); sin_a = np.sin(angle)
        e1_rot =  cos_a * e1 + sin_a * e2
        e2_rot = -sin_a * e1 + cos_a * e2
        e1, e2 = e1_rot, e2_rot

    v0 = C - a*e1 - b*e2
    v1 = C + a*e1 - b*e2
    v2 = C + a*e1 + b*e2
    v3 = C - a*e1 + b*e2

    vertices = np.array([v0, v1, v2, v3])
    faces    = np.array([[0, 1, 2], [0, 2, 3]])

    rect_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            rect_mesh.vectors[i, j] = vertices[f[j], :]

    rect_mesh.update_normals()
    rect_mesh.save(filename)


def stl_surface_normal(your_mesh):
    v0, v1, v2 = your_mesh.vectors[0]
    n = np.cross(v1 - v0, v2 - v0)
    n /= np.linalg.norm(n)
    return n


def surface_tangents(n):
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, n)) > 0.99:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = ref - np.dot(ref, n)*n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


def stl_surface_area(your_mesh):
    v0 = your_mesh.vectors[:, 0, :]
    v1 = your_mesh.vectors[:, 1, :]
    v2 = your_mesh.vectors[:, 2, :]
    return np.sum(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1))


def merge_meshes(mesh_list, merged_mesh_filename):
    for i, file in enumerate(mesh_list):
        m = mesh.Mesh.from_file(file)
        if i == 0:
            combined_data = m.data.copy()
        else:
            combined_data = np.concatenate([combined_data, m.data])
    combined_mesh = mesh.Mesh(combined_data.copy())
    combined_mesh.save(merged_mesh_filename)


def get_plane_corner(your_mesh):
    vertices         = your_mesh.vectors.reshape(-1, 3)
    unique_vertices  = np.unique(np.round(vertices, 6), axis=0)
    plane_normal     = stl_surface_normal(your_mesh)
    points_2d        = project_points_to_plane_2d(unique_vertices, plane_normal)
    hull             = ConvexHull(points_2d)
    return extract_corners_from_hull(points_2d[hull.vertices],
                                     unique_vertices[hull.vertices], 30)


def project_points_to_plane_2d(points, normal, origin=None):
    points = np.asarray(points, dtype=float)
    u, v   = surface_tangents(normal)
    if origin is None:
        origin = points.mean(axis=0)
    rel = points - np.asarray(origin, dtype=float)
    return np.column_stack([rel @ u, rel @ v])


def extract_corners_from_hull(hull_points_2d, hull_points_3d, angle_tol_deg=20):
    N = len(hull_points_2d)
    if N <= 3:
        return hull_points_3d
    corners   = []
    angle_tol = np.deg2rad(angle_tol_deg)
    for i in range(N):
        A = hull_points_2d[(i - 1) % N]
        B = hull_points_2d[i]
        C = hull_points_2d[(i + 1) % N]
        v1 = A - B; v1 /= np.linalg.norm(v1)
        v2 = C - B; v2 /= np.linalg.norm(v2)
        a = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        if np.abs(a - np.pi) > angle_tol:
            corners.append(hull_points_3d[i])
    return np.array(corners)


def get_plane_info(corners, mesh_obj):
    A, B, C, D = (np.asarray(corners[i]) for i in range(4))
    l1 = np.linalg.norm(A - B)
    l2 = np.linalg.norm(A - C)
    l3 = np.linalg.norm(A - D)

    t       = np.array([(B-A)/l1, (C-A)/l2, (D-A)/l3])
    centers = np.array([(B+A)/2,  (C+A)/2,  (D+A)/2])

    l_array = np.array([l1, l2, l3])
    i_max   = np.argmax(l_array)
    l_array = np.delete(l_array, i_max)
    t       = np.delete(t, i_max, axis=0)
    plane_center = centers[i_max]

    sort_index    = np.argsort(l_array)
    height        = l_array[sort_index[0]]
    width         = l_array[sort_index[1]]
    secondary_dir = t[sort_index[1]]
    third_dir     = t[sort_index[0]]
    main_dir      = stl_surface_normal(mesh_obj)

    return height, width, plane_center, main_dir, secondary_dir, third_dir


#######################################################################################
# Set unit to SI and apply angle calibration
spray_angle_commanded_deg = spray_angle_target_deg * calibration_factor
angle             = np.deg2rad(spray_angle_commanded_deg)
volume_flowrate   = volume_flowrate / 60 / 1000     # L/min -> m^3/s

print("="*64)
print(f"  Target spray angle  : {spray_angle_target_deg:.2f} deg")
print(f"  Calibration factor  : {calibration_factor:.4f}")
print(f"  Commanded angle     : {spray_angle_commanded_deg:.2f} deg")
print(f"  Edge-boost formul.  : {EDGE_BOOST_FORMULATION}")
print("="*64)

#-----------------------------------
# define mesh object
#--------------------------------------------------------------------
if inlet_plane_stl is None:
    inlet_plane_stl = path + 'rectangle.stl'
    rectangle_from_center_width_height(rectangle_plane['center'],
                                       rectangle_plane['width'],
                                       rectangle_plane['height'],
                                       rectangle_plane['normal'],
                                       rectangle_plane['angle'],
                                       inlet_plane_stl)

your_mesh = mesh.Mesh.from_file(inlet_plane_stl)
corners   = get_plane_corner(your_mesh)
spray_height, spray_width, spray_center, main_dir, secondary_dir, third_dir = \
    get_plane_info(corners, your_mesh)
area = stl_surface_area(your_mesh)

#--------------------------------------------------------------------
# Define particle size
#--------------------------------------------------------------------
if multi_inlet:
    d_p = min(d_p, spray_height / 2)
else:
    d_p = min(d_p, spray_height / 6)

inlet_velocity = volume_flowrate / area
npts           = int(angle / np.deg2rad(1))

#--------------------------------------------------------------------
# Generate point coordinates for velocity inlet mapping
#--------------------------------------------------------------------
if True:
    alpha = np.linspace(-angle/2, angle/2, npts)

    x1lin = np.zeros(npts)
    x1    = x1lin[:, None] * main_dir + spray_center*main_dir

    x2lin = np.linspace(-spray_width/2, spray_width/2, npts)
    x2    = x2lin[:, None] * secondary_dir + spray_center*secondary_dir

    x3lin = np.zeros(npts)
    x3    = x3lin[:, None] * third_dir + spray_center*third_dir

    xyz   = x1 + x2 + x3

    if multi_inlet:
        n_sub_inlets    = int(spray_width / (d_p*1.999))
        sub_inlet_width = spray_width / n_sub_inlets * 1.02

        sub_inlet_total_surface = n_sub_inlets * sub_inlet_width * spray_height
        correction_surface      = sub_inlet_total_surface / (spray_width*spray_height)

        x2lin            = np.linspace(-spray_width/2, spray_width/2, n_sub_inlets)
        x2_sub_inlets    = x2lin[:, None] * secondary_dir + spray_center*secondary_dir
        alpha_inlets     = np.linspace(-angle/2, angle/2, n_sub_inlets)

        center_sub_inlet = np.zeros(3)
        axis_sub_inlet   = np.zeros(3)
        file_name_list_sub_inlets = []
        for i in range(n_sub_inlets):
            center_sub_inlet[:] = x1[0,:] + x2_sub_inlets[i,:] + x3[0,:]
            axis_sub_inlet[:]   = main_dir + secondary_dir * np.tan(alpha_inlets[i])
            axis_sub_inlet      = axis_sub_inlet/np.linalg.norm(axis_sub_inlet)

            rectangle_from_center_width_height(center_sub_inlet,
                                               sub_inlet_width, spray_height,
                                               axis_sub_inlet, 0,
                                               f'{path}/sub_inlet_{i}.stl')
            file_name_list_sub_inlets.append(f'{path}/sub_inlet_{i}.stl')

    #----------------------------------------------------------------
    # Generate velocity for inlet mapping
    #----------------------------------------------------------------
    if sub_inlet_correction and multi_inlet:
        inlet_velocity = inlet_velocity/correction_surface

    # ---- Velocity formulation switch -------------------------------
    if EDGE_BOOST_FORMULATION:
        # Original (cos, tan): constant normal flux, edges go faster
        u1 = np.full(npts, inlet_velocity)
        u2 = np.full(npts, inlet_velocity) * np.tan(alpha)
    else:
        # Corrected (cos, sin): constant magnitude, true direction
        u1 = inlet_velocity * np.cos(alpha)
        u2 = inlet_velocity * np.sin(alpha)

    u1 = u1[:, None] * main_dir
    u2 = u2[:, None] * secondary_dir
    u3 = np.zeros((npts, 3))
    u_xyz = u1 + u2 + u3

    #----------------------------------------------------------------
    # save mapped velocity
    #----------------------------------------------------------------
    save_3D_field(points=xyz, vector_data=u_xyz,
                  vector_names=['velocity'],
                  file_name=path + 'inlet_velocity_vector.vtp')

    data   = np.column_stack([xyz[:,0], xyz[:,1], xyz[:,2],
                              u_xyz[:,0], u_xyz[:,1], u_xyz[:,2]])
    header = "pos_x,pos_y,pos_z,vel_x,vel_y,vel_z"
    np.savetxt(path+'flat_spray_velocity.csv', data,
               delimiter=',', comments='', header=header)

    #----------------------------------------------------------------
    # push to DIVE
    #----------------------------------------------------------------
    if upload_to_sim:
        from fisherman import (Project, Datatable, Settings,
                               OpenBoundary, OpenBoundaryDirection, OpenBoundaryType)

        project = Project()
        try:
            project = Project.get(name=project_name)[0]
        except Exception:
            project.add(name=project_name)

        try:
            simulation = project.get_simulations(name=sim_name)[0]
        except Exception:
            simulation = project.add_simulation(name=sim_name, description='')

        inlet_condition = OpenBoundary(
            io_direction = OpenBoundaryDirection.INLET,
            io_type      = OpenBoundaryType.RIEMANN_VELOCITY,
            profile      = Datatable(path+'flat_spray_velocity.csv'),
        )

        if multi_inlet:
            for i, inlet_file in enumerate(file_name_list_sub_inlets):
                inlet = simulation.add_open_boundary(
                    name          = f'sub_inlet_{i}',
                    filepath      = inlet_file,
                    open_boundary = inlet_condition,
                )
        else:
            inlet = simulation.add_open_boundary(
                name          = 'inlet',
                filepath      = f'{path}/rectangle.stl',
                open_boundary = inlet_condition,
            )

        settings = Settings(particle_diameter=float(d_p))
        simulation.update_settings(settings)
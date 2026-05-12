import os
import numpy as np
from stl import mesh    # pip install numpy-stl
from scipy.spatial import ConvexHull
import pyvista as pv
#######################################################################################
# READ ME
"""
FDX (fluidic-oscillator / sweep-jet) inlet emulation -- time-varying velocity version.

DIVE's open-boundary CSV accepts a `time` column, so we encode the FDX sweep
directly as a time-varying velocity vector at each spatial point on a small,
fixed inlet plane.

Sweep waveform (sinusoidal):
    alpha(t) = (alpha_max/2) * sin(2*pi*f*t)
    v(t)     = v_jet * [ cos(alpha(t)) * main_dir  +  sin(alpha(t)) * secondary_dir ]

CSV column order (matches the DIVE Velocity Distribution table):
    time, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z

All spatial points share the same velocity at a given time (the "jet"
sweeps as one stream, not as a fan).
"""
#######################################################################################
# REQUIRED PACKAGES
# pip install numpy-stl scipy numpy pyvista
# pip install dive-fisherman --extra-index-url https://pkgs.dev.azure.com/divesph/dive-public/_packaging/dive-registry/pypi/simple/
#######################################################################################
path = os.path.dirname(os.path.abspath(__file__))+'/'
#######################################################################################
project_name = 'FDX oscillating inlet'
sim_name     = 'FDX_retainer_60deg_70hz'

# ---- FDX physical parameters --------------------------------------------------
spray_angle        = 60           # peak-to-peak sweep angle [deg]
f_osc              = 70           # oscillation frequency    [Hz]
jet_velocity       = 22.0         # jet speed                [m/s]
total_duration     = 4.0          # how long the CSV covers  [s]

# ---- Inlet plane --------------------------------------------------------------
inlet_plane_stl = None  # auto-build if None
rectangle_plane = {
    'center': [-0.08,0.33,0.69],
    'width':  0.6e-3,             # 0.6 mm — along secondary direction (sweep plane)
    'height': 1.5e-3,             # 1.5 mm — along third direction (out of sweep plane)
    'normal': [0, -1, 0],          # jet emerges in +Y
    'angle':  0,
}

# ---- CSV resolution -----------------------------------------------------------
samples_per_period = 40           # time resolution per sweep period
n_grid             = 5            # spatial grid (n_grid x n_grid points)
sweep_sign         = +1           # flip to -1 if sweep goes the wrong way

# ---- Particle size (~6 SPH particles across the narrow dimension) -------------
d_p                = rectangle_plane['width'] / 6     # 0.6 mm / 6 = 100 µm

upload_to_sim      = True

#######################################################################################
# Helper functions (lifted from the original DIVE script)
#######################################################################################
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
    sort_index   = np.argsort(l_array)
    height = l_array[sort_index[0]]
    width  = l_array[sort_index[1]]
    secondary_dir = t[sort_index[1]]
    third_dir     = t[sort_index[0]]
    main_dir      = stl_surface_normal(mesh_obj)
    return height, width, plane_center, main_dir, secondary_dir, third_dir


#######################################################################################
# 1) Build / load the inlet plane
#######################################################################################
angle_max = np.deg2rad(spray_angle)

if inlet_plane_stl is None:
    inlet_plane_stl = path + 'rectangle.stl'
    rectangle_from_center_width_height(
        rectangle_plane['center'],
        rectangle_plane['width'],
        rectangle_plane['height'],
        rectangle_plane['normal'],
        rectangle_plane['angle'],
        inlet_plane_stl,
    )

your_mesh = mesh.Mesh.from_file(inlet_plane_stl)
corners   = get_plane_corner(your_mesh)
spray_height, spray_width, spray_center, main_dir, secondary_dir, third_dir = \
    get_plane_info(corners, your_mesh)
area = stl_surface_area(your_mesh)

# Particle size cap (single-inlet rule)
d_p = min(d_p, spray_height / 6)

#######################################################################################
# 2) Sanity prints
#######################################################################################
period      = 1.0 / f_osc
n_periods   = total_duration * f_osc
n_time      = max(int(np.ceil(n_periods * samples_per_period)), 100)
n_points    = n_grid * n_grid
n_rows      = n_time * n_points
mean_flow   = area * jet_velocity * np.mean(np.cos(np.linspace(-angle_max/2, angle_max/2, 1000)))
# (cos-averaged because the effective normal flux through the plane is v_jet*cos(alpha))

print("="*64)
print("FDX time-varying inlet -- summary")
print("="*64)
print(f"  Inlet area              : {area*1e6:.4f} mm^2")
print(f"  Jet velocity            : {jet_velocity:.2f} m/s")
print(f"  Peak vol. flow (alpha=0): {area*jet_velocity*60*1000:.4f} L/min")
print(f"  Mean vol. flow (sweep)  : {mean_flow*60*1000:.4f} L/min  (approx)")
print(f"  Sweep angle             : +/- {spray_angle/2:.1f} deg")
print(f"  Oscillation freq        : {f_osc:.1f} Hz  (T = {period*1e3:.3f} ms)")
print(f"  Duration                : {total_duration*1e3:.1f} ms ({n_periods:.1f} periods)")
print(f"  Time samples            : {n_time}")
print(f"  Spatial grid            : {n_grid} x {n_grid} = {n_points} points")
print(f"  CSV rows                : {n_rows}")
print(f"  Particle diameter       : {d_p*1e6:.1f} um")
print(f"  main_dir                : {main_dir}")
print(f"  secondary_dir           : {secondary_dir}  (sweep happens in this direction)")
print(f"  third_dir               : {third_dir}      (perpendicular to sweep)")
print("="*64)

#######################################################################################
# 3) Build spatial grid of points on the inlet plane
#######################################################################################
u_lin           = np.linspace(-spray_width /2, spray_width /2, n_grid)
v_lin           = np.linspace(-spray_height/2, spray_height/2, n_grid)
uu, vv          = np.meshgrid(u_lin, v_lin)
u_flat, v_flat  = uu.flatten(), vv.flatten()

xyz = (spray_center[None, :]
       + u_flat[:, None] * secondary_dir[None, :]
       + v_flat[:, None] * third_dir[None, :])     # shape (n_points, 3)

#######################################################################################
# 4) Build the time-varying velocity vector (same for every spatial point)
#######################################################################################
t_arr     = np.linspace(0.0, total_duration, n_time)
alpha_t   = sweep_sign * (angle_max / 2.0) * np.sin(2.0 * np.pi * f_osc * t_arr)

# Velocity components in world frame at each time
v_main    = jet_velocity * np.cos(alpha_t)        # along main_dir
v_secd    = jet_velocity * np.sin(alpha_t)        # along secondary_dir

# Build the full (n_time, 3) velocity-vs-time array
vel_t = (v_main[:, None] * main_dir[None, :]
         + v_secd[:, None] * secondary_dir[None, :])     # shape (n_time, 3)

#######################################################################################
# 5) Tile spatial grid x time into one big CSV
#       Row ordering: (t0, p0), (t0, p1), ..., (t0, pN), (t1, p0), ...
#    DIVE seems happy with arbitrary ordering; this is just the most straightforward.
#######################################################################################
T_rep  = np.repeat(t_arr,    n_points)              # (n_time*n_points,)
P_rep  = np.tile  (xyz,      (n_time, 1))           # (n_time*n_points, 3)
V_rep  = np.repeat(vel_t,    n_points, axis=0)      # (n_time*n_points, 3)

csv_data   = np.column_stack([T_rep,
                              P_rep[:, 0], P_rep[:, 1], P_rep[:, 2],
                              V_rep[:, 0], V_rep[:, 1], V_rep[:, 2]])
csv_header = "time,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z"

csv_path = path + 'fdx_velocity_timevarying.csv'
np.savetxt(csv_path, csv_data,
           delimiter=',', comments='', header=csv_header,
           fmt=['%.6e']*7)

print(f"\nWrote {csv_path}  ({csv_data.shape[0]} rows, "
      f"{os.path.getsize(csv_path)/1e6:.2f} MB)")

# Also dump a Paraview-friendly snapshot of vel(t=0) for visual sanity check
m_vtp = pv.PolyData(xyz)
m_vtp.point_data['velocity_t0'] = np.tile(vel_t[0], (xyz.shape[0], 1))
m_vtp.save(path + 'fdx_inlet_velocity_t0.vtp')

#######################################################################################
# 6) Upload to DIVE
#######################################################################################
if upload_to_sim:
    from fisherman import (Project, Datatable, Settings,
                           OpenBoundary, OpenBoundaryDirection, OpenBoundaryType,
                           FluidMaterial, AlertAction, AlertValueType,
                           RotatoryMotion)

    project = Project()
    try:
        project = Project.get(name=project_name)[0]
    except Exception:
        project.add(name=project_name)

    try:
        simulation = project.get_simulations(name=sim_name)[0]
    except Exception:
        simulation = project.add_simulation(name=sim_name, description='')

    # Water at 60 °C — density and kinematic viscosity from standard tables, SoS overridden
    water_60 = FluidMaterial(
        name           = 'Water 60°C',
        density        = 983.2,       # kg/m³
        viscosity      = 4.75e-7,     # kinematic viscosity [m²/s]
        isentropic_exp = 7,
        speed_of_sound = 200,         # m/s (user-specified)
    )
    simulation.use_material(water_60)

    inlet_condition = OpenBoundary(
        io_direction = OpenBoundaryDirection.INLET,
        io_type      = OpenBoundaryType.RIEMANN_VELOCITY,
        profile      = Datatable(csv_path),
    )

    inlet = simulation.add_open_boundary(
        name          = 'fdx_inlet',
        filepath      = inlet_plane_stl,
        open_boundary = inlet_condition,
    )

    gitter_motion = RotatoryMotion(
        name   = 'gitter_rotation',
        vel    = 0.62832,      # rad/s
        center = None,         # uses part geometry center
        axis   = [0, 0, 1],    # Z axis
    )
    simulation.add_wall_boundary(
        name                        = 'gitter',
        filepath                    = path + 'gitter.stl',
        resolution                  = 0.2,
        velocity_boundary_condition = 'moving',
        moving_boundary_condition   = gitter_motion,
    )

    settings = Settings(
        particle_diameter    = float(d_p),
        end_time             = total_duration,           # 0.1 s
        output_time_interval = total_duration / 1250,    # 1250 outputs
        surface_tension      = True,
        sigma                = 0.066,                    # N/m — surface tension coeff (water 60 °C)
    )
    simulation.update_settings(settings)

    simulation.set_alert(
        action     = AlertAction.STOP,
        value_type = AlertValueType.CREDITS,
        threshold  = 150,
    )

    simulation.start(confirm=False)

    print(f"\nUploaded to DIVE:")
    print(f"  Project    : {project_name}")
    print(f"  Simulation : {sim_name}")
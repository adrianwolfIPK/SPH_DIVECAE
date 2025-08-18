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

import math as mth

# extend class
class Project(Project):

    @property
    def name(self):
        return self.data.get("name")

    def print_summary(self):
        print(f"Project '{self.name}' with ID {self.id}")

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

def get_project_id(proj_name, projects):

    for p in projects:
        p.print_summary()

        if p.name == proj_name:
            print(f"-- Using '{proj_name}' Project -- \n")
            return p.id

def create_or_delete_sim(proj_name, sim_name):


    projects = Project.get(name=proj_name)
    id = get_project_id(proj_name, projects)

    project = Project(id)

    simulations = project.get_simulations()

    sim_found = False

    for s in simulations:
        print(s)

        if s.data.get("name") == sim_name:
            print(f"Simulation name is already used. Defaulting to using existing simulation: '{sim_name}'\n")
            simulation = s
            sim_found = True

            del_sim = get_user_input()

            if del_sim == "y" or del_sim == "yes":
                s.delete()
                sim_found = False
        
    if sim_found == False:
        print(f"Creating Simulation: {sim_name} in Project:{proj_name}\n")
        simulation = project.add_simulation(name=sim_name)

    return simulation

def get_sos(v_max=0, d_particle=1, p=0, rho=999, rho_ref=997, B=2.2e9, gamma=7, sigma=0, s_factor=1.5):
    """
    v_max       in  m/s\\
    d_particle  in  m\\
    p           in  N/m^2\\
    rho         in  kg/m^3\\
    sigma       in  N/m\\
    """
    factor = (10 + s_factor)
    sos_vmax = factor * v_max
    # sos_p = factor * mth.sqrt(2*p/rho) # given by knowledge base but seems unreasonably highly
    sos_p = mth.sqrt(B * gamma / rho_ref * (rho / rho_ref) ** (gamma-1))
    sos_st = 100 * mth.sqrt(2*mth.pi*sigma / (1.2*rho*d_particle))
    print(f"{sos_vmax = }, {sos_p = }, {sos_st = }")

    vals = [sos_vmax, sos_p, sos_st]
    max_val = max(vals)
    sum_val = mth.sqrt(sum(v**2 for v in vals if v >= 0.1 * max_val))
    
    print(f"The speed of sound should be chosen as {sum_val} m/s.")

    if sum_val > 1000:
        print(f"\n[WARNING] The upper limit of the artificial \
              speed of sound is 1000 m/s.")
        return 1000
    return sum_val

if __name__ == "__main__":
    get_sos(6.634, 9e-4, 3e5, 999)

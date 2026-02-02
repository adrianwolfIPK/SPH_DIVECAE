import csv
import os
import pandas as pd
import numpy as np

# -----------------------------
# USER INPUT
# -----------------------------
input_csv = r"C:\Users\adr61871\Desktop\fisherman_test\static\Nozzles\AX90_L.csv" # path to CSV
nozzle_name = r"AX90_L"

df = pd.read_csv(r"static\Nozzles\AX90_L.csv")
X = df.to_numpy()

xinit = 0.033
xOff = 0.239

# 90 profile in dive: 0.033, 1.1e-5, 0
# 90 vordermittelpunkt, oben rechts: 0.239, 0.565861, 0.785199
# 90 vordermittelpunkt, oben links: 0.239, 0.313861, 0.785199
'''
For AX90_L

xinit = 0.033
xOff = 0.253

    
For AX120_L

xinit = 0.042
xOff = 0.2535

(xOff-xinit, 0.566, 0.7032),
(xOff-xinit, 0.566, 0.6147),
(xOff-xinit, 0.314, 0.6147),
(xOff-xinit, 0.314, 0.7035)
'''

x1_e = (xOff-xinit, 0.565861, 0.785199)
x2_e = (xOff-xinit, 0.313861, 0.785199)

x_e = [x1_e, x2_e]

output_dir = "post/output"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# PROCESSING
# -----------------------------
with open(input_csv, newline="") as f:
    reader = list(csv.reader(f))
    header = reader[0]
    data = reader[1:]

for offset in x_e:
    # Compute adjusted positions for all rows as NumPy array
    data_array = np.array(data, dtype=float)  # shape: (num_rows, num_columns)
    
    # Adjust positions: columns 1,2,3 = x,y,z
    adjusted_positions = data_array[:, 1:4] + offset  # broadcasting
    
    # Pick the first row's final position
    x_final, y_final, z_final = adjusted_positions[0]

    output_file = os.path.join(
        output_dir,
        f"{nozzle_name}_x{x_final:.3f}_y{y_final:.3f}_z{z_final:.3f}.csv"
    )

    # Write the file
    with open(output_file, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)

        for row_idx, row in enumerate(data):
            time = row[0]
            pos_x, pos_y, pos_z = adjusted_positions[row_idx]
            
            vel_x = row[4]
            vel_y = row[5]
            vel_z = row[6]

            writer.writerow([time, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z])

    print(f"Written: {output_file}")

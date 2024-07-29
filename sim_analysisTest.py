import pandas as pd
import numpy as np
import ast
import os
import glob
import matplotlib.pyplot as plt

# Read the data from the file
input_data = pd.read_csv("input.dat", delimiter=' ', header=None).values.tolist()

# Extract the first element from the first row
trid = input_data[0][0]

# Import and transpose data from the first file in each folder
def import_and_transpose(file_path):
    data = pd.read_csv(file_path, header=None)
    return data.T.values.tolist()
if trid == '1':
    print('comes later')

else:
    # Convert the string to a list of integers, replacing '[' with '{' and ']' with '}' accordingly
    lat_size_str = input_data[1][0].replace('[', '{').replace(']', '}')
    lat_size = ast.literal_eval(lat_size_str.replace('{', '[').replace('}', ']'))

    # Flatten the input data from the third element onwards
    flattened_data = [item for sublist in input_data[2:] for item in sublist]
    float_data = map(float, flattened_data)

    # Assign the variables from the flattened list
    (J1, J2, tempInit, tempFinal, numberMeasurements, equilibration, stepsize, temperature, frames, decorr) = float_data
    numberMeasurements = numberMeasurements / decorr

    # Define the base directory
    base_directory = os.path.join(os.getcwd(), "Results")

    # Define string templates
    template = "J1={J1},J2={J2}/ResultPar/Result_L={L},T={T},h={h}/newpar_L={L},T={T},h={h}/newpar_L={L},T={T},h={h}.dat"
    mag_template = "J1={J1},J2={J2}/ResultPar/Result_L={L},T={T},h={h}/avgmag_L={L},T={T},h={h}/avgmag_L={L},T={T},h={h}.dat"
    energ_template = "J1={J1},J2={J2}/ResultPar/Result_L={L},T={T},h={h}/avgE_L={L},T={T},h={h}/avgE_L={L},T={T},h={h}.dat"

    # Define functions to format the strings
    def param_string(J1, J, L, Jt, h):
        return template.format(J1=f"{J1:.1f}", J2=f"{J:.3f}", L=L, T=f"{Jt:.3f}", h=f"{h:.3f}")

    def mag_param_string(J1, J, L, Jt, h):
        return mag_template.format(J1=f"{J1:.1f}", J2=f"{J:.3f}", L=L, T=f"{Jt:.3f}", h=f"{h:.3f}")

    def energ_param_string(J1, J, L, Jt, h):
        return energ_template.format(J1=f"{J1:.1f}", J2=f"{J:.3f}", L=L, T=f"{Jt:.3f}", h=f"{h:.3f}")
    
    Jmax = 1.0
    Jini = 0.000
    Jstep = 0.500
    
    # Generate the range of values
    Jrange = np.arange(Jini, Jmax + Jstep, Jstep)

    # Format the numbers to have three decimal places
    Jrange_formatted = [f"{j:.3f}" for j in Jrange]

    Tdata = [f"{t:.3f}" for t in np.arange(tempInit, tempFinal- stepsize, stepsize)]

    folders = [
    [
        [glob.glob(os.path.join(base_directory, param_string(J1, J*Jstep + Jini, lat_size[l], J*Jstep + Jini, t)))
            for t in np.arange(tempInit, tempFinal-stepsize, stepsize)
        ]
        for l in range(len(lat_size))
    ]
    for J in range(1, 2)
    ]

    foldersMag = [
    [
        [
            glob.glob(os.path.join(base_directory, mag_param_string(J1, J*Jstep + Jini, lat_size[l], J*Jstep + Jini, t)))
            for t in np.arange(tempInit, tempFinal, stepsize)
        ]
        for l in range(len(lat_size))
    ]
    for J in range(1, 2)
    ]

    foldersEnerg = [
    [
        [
            glob.glob(os.path.join(base_directory, energ_param_string(J1, J*Jstep + Jini, lat_size[l], J*Jstep + Jini, t)))
            for t in np.arange(tempInit, tempFinal, stepsize)
        ]
        for l in range(len(lat_size))
    ]
    for J in range(1, 2)
    ]

    # Determine the sizes
    size_folders = int(len(folders[0]))
    size_file_names = int(len(folders[0][0]))

    data_list_test = [
    [
        [
            import_and_transpose(folders[j][l][k][0])
            for k in range(size_file_names)
        ]
        for l in range(size_folders)
    ]
    for j in range(len(folders))
    ]

    # data_list_energ = [
    # [
    #     [
    #         import_and_transpose(foldersEnerg[j][l][k][0])
    #         for k in range(size_file_names)
    #     ]
    #     for l in range(size_folders)
    # ]
    # for j in range(len(folders))
    # ]

    # data_list_mag = [
    # [
    #     [
    #         import_and_transpose(foldersMag[j][l][k][0])
    #         for k in range(size_file_names)
    #     ]
    #     for l in range(size_folders)
    # ]
    # for j in range(len(folders))
    # ]

    meanNewParData = np.zeros((int(Jmax), size_folders, size_file_names))
    meanNewParData2 = np.zeros((int(Jmax), size_folders, size_file_names))
    meanNewParData4 = np.zeros((int(Jmax), size_folders, size_file_names))
    errNewParData = np.zeros((int(Jmax), size_folders, size_file_names))
    dataPlot = np.zeros((int(Jmax), size_folders, size_file_names), dtype=object)
    binderAll = np.zeros((int(Jmax), size_folders, size_file_names))
    dataBinderPlot = np.zeros((int(Jmax), size_folders, size_file_names), dtype=object)

    for j in range(int(Jmax)):
        for l in range(size_folders):
            for t in range(size_file_names):
                abs_data = np.abs(data_list_test[j][l][t][0])
                meanNewParData[j][l][t] = np.mean(abs_data)
                meanNewParData2[j][l][t] = np.mean(abs_data**2)
                meanNewParData4[j][l][t] = np.mean(abs_data**4)
                errNewParData[j][l][t] = np.std(abs_data) / np.sqrt(numberMeasurements)
                dataPlot[j][l][t] = (meanNewParData[j][l][t], errNewParData[j][l][t])
                binderAll[j][l][t] = 1 - meanNewParData4[j][l][t] / (3 * meanNewParData2[j][l][t]**2)
                dataBinderPlot[j][l][t] = (binderAll[j][l][t], errNewParData[j][l][t])
    
    for j in range(int(Jmax)):
        plt.figure(figsize=(8, 6))
        
        for l in range(size_folders):
            data = np.array([x for x in dataPlot[j, l, :]], dtype=object)
            means = np.array([x[0] for x in data], dtype=float)
            errors = np.array([x[1] for x in data], dtype=float)

            plt.errorbar(Tdata, means, yerr=errors, label=f'L={lat_size[l]}', fmt='-o')
        
        j_value = j * Jstep + Jini
        plt.title(f"New Order parameter for different lattice sizes, with J2={J2}, T={J2}")
        plt.xlabel(r"Magnetic field $H$")
        plt.ylabel(r"$\langle G \rangle$")
        plt.legend()
        plt.grid(True)
        plt.show()
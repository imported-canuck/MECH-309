import numpy as np
from matplotlib import pyplot as plt

def load_calibration_data(filename="calibrationData.csv"):
    """
    Load hot-wire calibration data from CSV file.

    v_data : np.ndarray
        Velocity measurements [m/s]
    y_data : np.ndarray
        Output voltage squared measurements [V^2]
    """

    calibrationData = np.loadtxt(
        filename,
        dtype=float,
        delimiter=",",
        skiprows=1,
        usecols=(0, 1)
    )

    v_data = calibrationData[:, 0]
    y_data = calibrationData[:, 1]

    return v_data, y_data

def GN_solve(v_data, y_data):
    
    pass

if __name__ == "__main__":
    # Load data
    v_data, y_data = load_calibration_data()

    # Plot calibration data
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5, forward=True)
    ax.set_title(r"Hot-Wire Anemometer - Calibration Data")
    ax.set_xlabel(r"Velocity, $v$ [$m/s$]")
    ax.set_ylabel(r"Output Voltage Squared, $y$ [$V^2$]")
    ax.plot(v_data, y_data, "o", label="Calibration Data")
    ax.legend(loc="upper left")
    plt.show()
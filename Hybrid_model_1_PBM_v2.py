import numpy as np


def main(pos_vel, dt):
    # Parameters #
    # au = 149597870700   Distance by astronomical units (m)
    # ve = 149597870700 / 86400   Velocity by astronomical units/day (m/s)
    # G = 6.67430e-11  Gravitational constant (m^3/(kg*s^2))
    G = 6.67430e-11 * 86400**2 * 1.9885e30 / 149597870700**3  # G constant by au (au^3/(M_sun*day^2))
    dt = dt  # 1/24/60*10  #/86400  # Time step
    n = 9  # Number of celestial bodies

    # Dictionary with the celestial bodies in the solar system.
    # Keys = name, values = sphere-function from vpython with different attributes
    solar_system = [[0, 0, 1],
                    [0, 0, 3.302e23/1.9885e30],
                    [0, 0, 4.8685e24/1.9885e30],
                    [0, 0, 5.97219e24/1.9885e30],
                    [0, 0, 6.4171e23/1.9885e30],
                    [0, 0, 1.89818722e27/1.9885e30],
                    [0, 0, 5.6834e26/1.9885e30],
                    [0, 0, 8.6813e25/1.9885e30],
                    [0, 0, 1.02409e26/1.9885e30]
                    ]

    for i in range(1, n, 1):
        solar_system[i][0] = np.array(pos_vel[i-1][0:3])
        solar_system[i][1] = np.array(pos_vel[i-1][3:6])

    # Updates velocity in the Solar system
    def update_velocity(i):
        j = 0
        gforce_sum = [0, 0, 0]
        while j <= n - 1:
            if j == i:
                j += 1
            else:
                gforce = solar_system[j][2] * (solar_system[i][0] - solar_system[j][0]) \
                         / (np.linalg.norm(solar_system[i][0] - solar_system[j][0])) ** 3
                gforce_sum += gforce
                j += 1
        solar_system[i][1] = solar_system[i][1] - dt * G * gforce_sum

    # Updates position in the solar system
    def update_position(i):
        solar_system[i][0] = solar_system[i][0] + dt * solar_system[i][1]

    # Calculates and returns total mechanical energy of the Solar system
    def energy():
        i = 0
        j = 0
        total_mec_energy = 0
        while i <= n - 1:
            gforce_sum = 0
            j = i + 1
            while j <= n - 1:
                gforce = solar_system[i][2] * solar_system[j][2] /\
                         (np.linalg.norm(solar_system[i][0] - solar_system[j][0]))
                gforce_sum += gforce
                j += 1
            mec_energy = solar_system[i][2] * np.linalg.norm(solar_system[i][1]) / 2 - G * gforce_sum
            total_mec_energy += mec_energy
            i += 1
        return total_mec_energy

    i = 1
    while i <= n - 1:
        update_velocity(i)
        update_position(i)
        H = energy()
        i += 1

    data = [x[:2] for x in solar_system[1:]]
    data = [i for x in data for j in x for i in j]
    return data, H


if __name__ == '__main__':
    main(pos_vel, dt)

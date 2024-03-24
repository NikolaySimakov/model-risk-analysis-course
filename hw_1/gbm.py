import numpy as np


def simulateGBM(
    spotPrice: float,
    volatility: float,
    drift: float,
    maturity: float,
    stepNumber: int,
    simulationNumber: int
) -> np.array:
    """
    Input parameters:
    spotPrice - initial asset price
    volatility - asset volatility
    drift - asset druft
    maturity - end of simulation in years
    stepNumber - partition time steps
    simulationNumber - number of paths to simulate

    Returns: 
    np.array of simulated paths
    """

    np.random.seed(52)
    dt = maturity / stepNumber
    driftGBM = (drift - 0.5 * volatility ** 2) * dt
    standartNormalSimulation = np.random.normal(
        0, 1, (stepNumber, simulationNumber))
    diffusionGBM = volatility * standartNormalSimulation * np.sqrt(dt)

    processValues = np.zeros((stepNumber, simulationNumber))
    processValues[0, :] = spotPrice
    for i in range(1, stepNumber):
        processValues[i, :] = processValues[i - 1, :] * \
            np.exp(driftGBM + diffusionGBM[i - 1, :])

    return processValues

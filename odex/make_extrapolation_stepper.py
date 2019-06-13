import numpy as np
import sympy
from .extrapolation_stepper import ExtrapolationStepper
from .compute_rextrap_weights import compute_rextrap_weights
from .gbs import GBS


def make_extrap_config(config=None):
    """Create the step counts and weights for a given order and number of cores
       :param system: callable ODE to time step, where y\'=system(t,state)
       :param state: state of the system
       :param config: dictionary specifying order and number of cores
       :param parallel: boolean to select whether to distribute across cores
     """
    if config is None:
        config = {'order':8, 'cores':3}

    order = config['order']
    cores = config['cores']
    steps = None

    if order == 8:
        if cores == 3:
            steps = [2, 16, 18, 20]
            weights = compute_rextrap_weights(steps)
            weights = np.array(weights).astype(np.float64)
            isbn = 0.5799

        elif cores == 6:
            ndep = [2, 4, 6, 10]
            nfree = [8, 12, 14, 16, 18, 20, 22]
            cfree = ['269/95360', '13805/611712', '83/1314', '3607/16544',
                     '2196/619', '-35962/2395', '7352/605']
            cdep = compute_rextrap_weights(ndep, nfree, cfree)
            cdep = np.array(cdep).astype(np.float64)

            cfree = sympy.sympify(cfree)
            cfree = np.array(cfree).astype(np.float64)

            steps = np.append(ndep, nfree)
            weights = np.append(cdep,cfree)
            isbn = 0.7695

        elif cores == 8:
            ndep = [2, 26, 28, 30]
            nfree = list(range(4,26,2))
            cfree = ['10827/755105792', '7807/65552384', '16757/38105088', '9667/8105984',
                     '1229/449920', '10649/1863936', '6499/565376', '12003/517120',
                     '5003/101760', '3767/32472', '2843/8591']
            cdep = compute_rextrap_weights(ndep, nfree, cfree)
            cdep = np.array(cdep).astype(np.float64)

            cfree = sympy.sympify(cfree)
            cfree = np.array(cfree).astype(np.float64)

            steps = np.append(ndep, nfree)
            weights = np.append(cdep,cfree)
            isbn = 0.8176

    elif order == 12:
        if cores == 4:
            steps = [2, 8, 12, 14, 16, 20]
            weights = compute_rextrap_weights(steps)
            weights = np.array(weights).astype(np.float64)
            isbn = 0.4515

        elif cores == 8:
            ndep = [2, 8, 10, 16, 24, 26]
            nfree = [4, 6, 12, 14, 18, 20, 22, 28, 30]
            cfree = ['12985/994150711296', '3295/1296039936', '2521/8515584',
                     '1349/1959936', '11223/1226368', '5711/69600', '6007/2478',
                     '-1338112/5553', '50764/475']
            cdep = compute_rextrap_weights(ndep, nfree, cfree)
            cdep = np.array(cdep).astype(np.float64)

            cfree = sympy.sympify(cfree)
            cfree = np.array(cfree).astype(np.float64)

            steps = np.append(ndep, nfree)
            weights = np.append(cdep,cfree)
            isbn = 0.7116

    elif order == 16:
        if cores == 5:
            steps = [2, 8, 10, 12, 14, 16, 18, 22]
            weights = compute_rextrap_weights(steps)
            weights = np.array(weights).astype(np.float64)
            isbn = 0.4162

    if steps is None:
        raise ValueError('Unknown Stepper Configuration')

    steppers = [GBS(step) for step in steps]
    return steppers, steps, weights, isbn


def make_extrapolation_stepper(system, state, config=None, parallel=True):
    """Construct an ExtrapolationStepper with optional config
       :param system: callable ODE to time step, where y\'=system(t,state)
       :param state: state of the system
       :param config: dictionary specifying order and number of cores
       :param parallel: boolean to select whether to distribute across cores
     """
    steppers, steps, weights, isbn = make_extrap_config(config)
    return ExtrapolationStepper(steppers, steps, weights, system, state, isbn=isbn, parallel=parallel)



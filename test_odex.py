import odex
import time
import numpy as np
import numpy.fft
import sympy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as anim


def make_extrap_config(label=None):
    if label is None:
        label = 'GBS_8,3'

    if label=='GBS_8,3':
        steps = [2, 16, 18, 20]
        weights = odex.compute_rextrap_weights(steps)
        weights = np.array(weights).astype(np.float64)
        steppers = [odex.GBS(step) for step in steps]
        num_cores = 3
        isb = 0.5799

    elif label=='GBS_12,4':
        steps = [2, 8, 12, 14, 16, 20]
        weights = odex.compute_rextrap_weights(steps)
        weights = np.array(weights).astype(np.float64)
        steppers = [odex.GBS(step) for step in steps]
        num_cores = 4
        isb = 0.4515

    elif label=='GBS_16,5':
        steps = [2, 8, 10, 12, 14, 16, 18, 22]
        weights = odex.compute_rextrap_weights(steps)
        weights = np.array(weights).astype(np.float64)
        steppers = [odex.GBS(step) for step in steps]
        num_cores = 5
        isb = 0.4162

    elif label=='GBS_8,6':
        ndep = [2, 4, 6, 10]
        nfree = [8, 12, 14, 16, 18, 20, 22]
        cfree = ['269/95360', '13805/611712', '83/1314', '3607/16544',
                 '2196/619', '-35962/2395', '7352/605']
        cdep = odex.compute_rextrap_weights(ndep, nfree, cfree)
        cdep = np.array(cdep).astype(np.float64)

        cfree = sympy.sympify(cfree)
        cfree = np.array(cfree).astype(np.float64)

        steps = np.append(ndep, nfree)
        weights = np.append(cdep,cfree)
        steppers = [odex.GBS(step) for step in steps]
        num_cores = 6
        isb = 0.7695

    elif label=='GBS_12,8':
        ndep = [2, 8, 10, 16, 24, 26]
        nfree = [4, 6, 12, 14, 18, 20, 22, 28, 30]
        cfree = ['12985/994150711296', '3295/1296039936', '2521/8515584',
                 '1349/1959936', '11223/1226368', '5711/69600', '6007/2478',
                 '-1338112/5553', '50764/475']
        cdep = odex.compute_rextrap_weights(ndep, nfree, cfree)
        cdep = np.array(cdep).astype(np.float64)

        cfree = sympy.sympify(cfree)
        cfree = np.array(cfree).astype(np.float64)

        steps = np.append(ndep, nfree)
        weights = np.append(cdep,cfree)
        steppers = [odex.GBS(step) for step in steps]
        num_cores = 8
        isb = 0.7116

    else:
        raise ValueError('Unknown Stepper Label')
        
    return steppers, steps, weights, num_cores, isb


def run_odex_simple(config=None,fn_eval_time=0):
    """Test the extrapolation scheme running on a given number of cores.
       :param fn_eval_time: Approximate time each ODE function evaluation
                            should take to simulate different scale problems
    """

    # ODE: y' = system(t,y)
    def system(t,y):
        if fn_eval_time > 0:
            time.sleep(fn_eval_time)
        return y

    # Initial conditions and number of steps to take
    y0 = 1
    t0 = 0
    t1 = 2
    n  = 2000
    dt = float(t1-t0)/n

    # Construct the extrapolation stepper
    steppers, steps, weights, num_cores, isb = make_extrap_config(config)
    stepper = odex.ExtrapolationStepper(steppers, steps, weights, system, y0, num_cores=num_cores)

    print('odex: simple ode on num_cores == {}...'.format(num_cores))

    # Solve the system, profiling
    start = time.time()
    yn = stepper.step(y0, t0, dt, n)
    duration = time.time()-start
    stepper.join()

    # Compute the error, print the results
    error = yn[-1]-np.exp(t1)
    print('  error: {}'.format(error))
    print('  duration: {}'.format(duration))
    assert abs(error) < 2e-10

    iters = 100
    mean_eval_time = 0
    for ii in range(iters):
        start = time.time()
        system(t0,y0)
        mean_eval_time += time.time()-start
    mean_eval_time /= iters

    estimated_duration = n*np.sum(np.array(steps)+1)*mean_eval_time
    print('  estimated duration: {}'.format(estimated_duration))
    print('  efficiency: {0:.2f}%'.format(100*estimated_duration/duration))

    return duration


def run_odex_convection_2d(config, do_plot, outfile=None):
    steppers, steps, weights, num_cores, isb = make_extrap_config(config)

    print('odex: 2D convection on num_cores == {}...'.format(num_cores))

    # PDE initial data
    npoints = 64
    xx = np.linspace(0,npoints-1,npoints, dtype=np.float64)
    xx,yy = np.meshgrid(xx,xx)
    u0 = np.exp(-60*((xx/npoints-.5)**2+(yy/npoints-.5)**2))

    # PDE and solver parameters
    c  = [0.5, 0.25]     # Wave speed
    k  = 1.              # Unit grid spacing
    t0 = 0.              # Simulation start time
    nsteps = 2048        # Number of time steps
    spectral = True

    # Set up maximum time step size
    hmax = k*(max(steps)+1)*isb/(2*max(c))
    if spectral: hmax = hmax/np.pi
    dt = hmax*.99

    # Movie writer setup
    if outfile is not None:
        moviewriter = anim.FFMpegWriter(fps=30)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        moviewriter.setup(fig, outfile, dpi=100)

        def observer(t,state):
            ax.clear()
            ax.plot_wireframe(xx, yy, state)
            ax.set_zlim(0,1)
            moviewriter.grab_frame()
    else:
        observer = None

    # Spectral gradient helper
    def islope():
        n   = len(u0);
        kn  = 2.*np.pi/n/k;
        iw = 1j*kn*np.array(list(range(0, int(n/2)+1)) + list(range(-int(n/2)+1,0)))
        return np.meshgrid(iw, iw)
    ikx, iky = islope()

    def spectral_gradient(u, k):
        U = np.fft.fft2(u)
        return np.real(np.fft.ifft2(ikx*U)), np.real(np.fft.ifft2(iky*U))

    def central_difference(u, k):
        n  = len(u)
        ux = np.empty(np.shape(u))
        uy = np.empty(np.shape(u))

        ux[:,1:n-1] = (u[:,2:n]-u[:,:n-2])/(2*k)
        ux[:,0    ] = (u[:,1  ]-u[:, n-1])/(2*k)
        ux[:,n-1  ] = (u[:,0  ]-u[:, n-2])/(2*k)

        uy[1:n-1,:] = (u[2:n,:]-u[:n-2,:])/(2*k)
        uy[0,    :] = (u[1,  :]-u[ n-1,:])/(2*k)
        uy[n-1,  :] = (u[0,  :]-u[ n-2,:])/(2*k)

        return ux, uy

    if spectral:
        derivfn = spectral_gradient
    else:
        derivfn = central_difference

    def gradient(u, k):
        return derivfn(u, k)

    # PDE system: transport
    def system(t, u):
        ux,uy = gradient(u, k)
        ut = -c[0]*ux-c[1]*uy
        return ut

    # Construct the extrapolation stepper
    stepper = odex.ExtrapolationStepper(steppers, steps, weights, system, u0, num_cores=num_cores)

    # Step the system once to ensure one-time setup completes before profiling
    stepper.step(u0, t0, dt, 1, dense_output=None, observer=None)

    # Solve the system, profiling
    start = time.time()
    un = stepper.step(u0, t0, dt, nsteps, dense_output=None, observer=observer)
    duration = time.time()-start
    stepper.join()

    if outfile is not None:
        moviewriter.finish()
        plt.close()

    # Compute the error, print the results
    print('  duration: {}'.format(duration))

    # Compute the mean evaluation time of the PDE system
    iters = 100
    mean_eval_time = 0
    for ii in range(iters):
        start = time.time()
        system(t0,u0)
        mean_eval_time += time.time()-start
    mean_eval_time /= iters

    # Print the profiling results
    estimated_duration = nsteps*np.sum(np.array(steps)+1)*mean_eval_time
    print('  mean eval duration: {}'.format(mean_eval_time))
    print('  estimated duration: {}'.format(estimated_duration))
    print('  efficiency: {0:.2f}%'.format(100*estimated_duration/duration))

    # Plot initial and final transport results
    if do_plot:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_wireframe(xx, yy, u0)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_wireframe(xx, yy, un)

        plt.grid()
        plt.show()

    return duration


def test_odex_convection_2d(plot_only=False):
    print('')
    duration = run_odex_convection_2d('GBS_12,4',plot_only)
    print('')
    print('Solver Duration: {}'.format(duration))
    print('')


def sanity_check():
    run_odex_simple('GBS_8,3')
    run_odex_simple('GBS_12,4')
    run_odex_simple('GBS_16,5')
    run_odex_simple('GBS_8,6')
    run_odex_simple('GBS_12,8')


def main():
    sanity_check()
    test_odex_convection_2d(True)


if __name__=='__main__':
    main()


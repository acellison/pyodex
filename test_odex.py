import odex
import time
import numpy as np
import numpy.fft
import sympy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as anim


def run_odex_simple(config, parallel=True):
    """Test the extrapolation scheme running on a given number of cores.
    """

    # ODE: y' = system(t,y)
    def system(t,y):
        return y

    # Initial conditions and number of steps to take
    y0 = 1.
    t0 = 0.
    t1 = 2.
    n  = 20
    dt = float(t1-t0)/n

    # Construct the extrapolation stepper
    stepper = odex.make_extrapolation_stepper(system, y0, config=config, parallel=parallel)

    print('Testing GBS_{{{},{}}} on the ODE y\'=y...'.format(stepper.order,stepper.ncores))

    # Solve the system, profiling
    start = time.time()
    yn = stepper.step(y0, t0, dt, n)
    duration = time.time()-start
    stepper.join()

    # Compute the error, print the results
    error = yn[-1]-np.exp(t1)
    print('  error: {}'.format(error))
    print('  duration: {}'.format(duration))
    assert abs(error) < 2.1e-12

    iters = 100
    mean_eval_time = 0
    for ii in range(iters):
        start = time.time()
        system(t0,y0)
        mean_eval_time += time.time()-start
    mean_eval_time /= iters

    estimated_duration = n*np.sum(np.array(stepper.stepcounts)+1)*mean_eval_time
    print('  estimated duration: {}'.format(estimated_duration))
    print('  efficiency: {0:.2f}%'.format(100*estimated_duration/duration))

    return duration


def run_odex_convection_2d(config, do_plot, outfile=None):
    print('odex: 2D convection...')

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
    stepper = odex.make_extrapolation_stepper(system, u0, config=config, parallel=True)

    # Set up maximum time step size
    hmax = k*(max(stepper.stepcounts)+1)*stepper.isbn/(2*max(c))
    if spectral: hmax = hmax/np.pi
    dt = hmax*.99

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
    estimated_duration = nsteps*np.sum(np.array(stepper.stepcounts)+1)*mean_eval_time
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


def test_odex_convection_2d(do_plot,filepath=None):
    print('')
    config = {'order':8, 'cores':8}
    duration = run_odex_convection_2d(config,do_plot,filepath)
    print('')
    print('Solver Duration: {}'.format(duration))
    print('')


def sanity_check():
    parallels = [True, False]
    for parallel in parallels:
        run_odex_simple({'order':8, 'cores':3},parallel)
        run_odex_simple({'order':8, 'cores':6},parallel)
        run_odex_simple({'order':8, 'cores':8},parallel)
        run_odex_simple({'order':12,'cores':4},parallel)
        run_odex_simple({'order':12,'cores':8},parallel)
        run_odex_simple({'order':16,'cores':5},parallel)


def main():
    sanity_check()
    filepath = None
    test_odex_convection_2d(do_plot=True,filepath=filepath)


if __name__=='__main__':
    main()


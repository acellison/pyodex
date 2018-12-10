import odex
import time
import numpy as np
import numpy.fft
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as anim

import cProfile
import pstats


def make_extrap_config():
    """8th order GBS-style extrapolation configuration.  ISO~=0.85"""
    steps = [4,  6,  8,  10,  12,  14,  16,  18,  20,  22,  24,  26,  2,  28,  30,  32]
    weights = [
        1.090847108057240618370944e-05,
        9.090694473649409970087559e-05,
        3.313334792337303976415830e-04,
        8.698152905260451896868323e-04,
        1.923504968239386162670579e-03,
        3.871085176250120205021510e-03,
        7.433942971759397583264573e-03,
        1.411088446832058258817444e-02,
        2.735278467616730768696698e-02,
        5.626374264639865535597707e-02,
        1.299352559996892475524533e-01,
        3.644164080047659348693401e-01,
        -5.382343886140006361529231e-07,
        7.652853451659328953837758e+00,
        -2.782569415349332331288679e+01,
        2.056623066697121515744584e+01
      ]
    steppers = [odex.GBS(step) for step in steps]
    return steppers, steps, weights


def run_odex_simple(num_cores, fn_eval_time=0):
    """Test the extrapolation scheme running on a given number of cores.
       :param num_cores: Number of cores on which to run the scheme
       :param fn_eval_time: Approximate time each ODE function evaluation
                            should take to simulate different scale problems
    """
    print('odex: simple ode on num_cores == {}...'.format(num_cores))

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
    steppers, steps, weights = make_extrap_config()
    stepper = odex.ExtrapolationStepper(steppers, steps, weights, system, y0, num_cores=num_cores)

    # Solve the system, profiling
    start = time.time()
    yn = stepper.step(y0, t0, dt, n)
    duration = time.time()-start
    stepper.join()

    # Compute the error, print the results
    error = yn[-1]-np.exp(t1)
    print('  error: {}'.format(error))
    print('  duration: {}'.format(duration))
    assert abs(error) < 1e-10

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


def test_odex_simple():
    fn_eval_time = 4e-5
    num_cores = [1, 2, 4, 8]
    durations = [run_odex_simple(nc, fn_eval_time) for nc in num_cores]
    print('')
    for ii in range(len(num_cores)):
        print('Solver Duration ({} Thread{}): {}'.format(num_cores[ii], ' ' if num_cores[ii] == 1 else 's', durations[ii]))
    print('')


def run_odex_convection(num_cores, do_plot):
    print('odex: convection on num_cores == {}...'.format(num_cores))

    npoints = 4096
    xgrid = np.linspace(0,npoints-1,npoints, dtype=np.float64)
    u0 = np.exp(-200*(xgrid/npoints-.5)**2);   # Initial data
    c  = 1.  # Wave speed
    k  = 1.  # Unit grid spacing
    t0 = 0.
    t1 = npoints
    n  = npoints//4
    dt = float(t1-t0)/n

    def islope():
        n  = len(u0);
        kn = 2.*np.pi/n/k;
        return 1j*kn*np.array(list(range(0, int(n/2)+1)) + list(range(-int(n/2)+1,0)))
    ik = islope()

    def specder(u, k):
        return np.real(sp.ifft(ik*sp.fft(u)));

    def system(t, u):
        ux = specder(u, k)
        return -c*ux

    # Construct the extrapolation stepper
    steppers, steps, weights = make_extrap_config()
    stepper = odex.ExtrapolationStepper(steppers, steps, weights, system, u0, num_cores=num_cores)

    # Solve the system, profiling
    start = time.time()
    un = stepper.step(u0, t0, dt, n)
    duration = time.time()-start
    stepper.join()

    # Compute the error, print the results
    print('  duration: {}'.format(duration))

    iters = 200
    mean_eval_time = 0
    for ii in range(iters):
        start = time.time()
        system(t0,u0)
        mean_eval_time += time.time()-start
    mean_eval_time /= iters

    estimated_duration = n*np.sum(np.array(steps)+1)*mean_eval_time
    print('  mean eval duration: {}'.format(mean_eval_time))
    print('  estimated duration: {}'.format(estimated_duration))
    print('  efficiency: {0:.2f}%'.format(100*estimated_duration/duration))

    if do_plot:
        plt.plot(xgrid, u0)
        inds = np.linspace(0,n,65, dtype=int)
        inds = inds[0:len(inds)-1]
        [plt.plot(xgrid, un[ind,:]) for ind in inds]
        plt.grid()
        plt.show()

    return duration


def test_odex_convection(plot_only=False):
    # plot
    if plot_only:
        num_cores = [4]
    else:
        num_cores = range(2,9)

    durations = [run_odex_convection(nc, plot_only) for nc in num_cores]
    print('')
    for ii in range(len(num_cores)):
        print('Solver Duration ({} Thread{}): {}'.format(num_cores[ii], ' ' if num_cores[ii] == 1 else 's', durations[ii]))
    print('')


def profile_odex_convection(run):
    cpus = [1, 2, 4, 8]
    calls = []
    filenames = []
    for cpu in cpus:
        calls.append('run_odex_convection({},False)'.format(cpu))
        filenames.append('conv_{}cpu.stats'.format(cpu))

    if run:
        for call, filename in zip(calls, filenames):
            cProfile.run(call, filename)

    for filename in filenames:
        p = pstats.Stats(filename)
        p.strip_dirs().sort_stats('cumtime').print_stats()


def run_odex_convection_2d(num_cores, do_plot, outfile=None):
    print('odex: 2D convection on num_cores == {}...'.format(num_cores))

    # PDE initial data
    npoints = 256
    xx = np.linspace(0,npoints-1,npoints, dtype=np.float64)
    xx,yy = np.meshgrid(xx,xx)
    u0 = np.exp(-60*((xx/npoints-.5)**2+(yy/npoints-.5)**2))

    # PDE and solver parameters
    c  = [0.5, 0.25]     # Wave speed
    k  = 1.              # Unit grid spacing
    t0 = 0.              # Simulation start time
    if outfile:
        t1 = 4*npoints   # Simulation end time
    else:
        t1 = npoints//2  # Simulation end time
    n  = t1//4           # Number of points
    dt = float(t1-t0)/n  # Time step size

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

    # Spatial derivative helper
    def islope():
        n  = len(u0);
        kn = 2.*np.pi/n/k;
        iw = 1j*kn*np.array(list(range(0, int(n/2)+1)) + list(range(-int(n/2)+1,0)))
        return np.meshgrid(iw,iw)
    ikx,iky = islope()

    def gradient(u, k):
        U = np.fft.fft2(u)
        return np.real(np.fft.ifft2(ikx*U)), np.real(np.fft.ifft2(iky*U))

    # PDE system: transport
    def system(t, u):
        ux,uy = gradient(u, k)
        ut = -c[0]*ux-c[1]*uy
        return ut

    # Construct the extrapolation stepper
    steppers, steps, weights = make_extrap_config()
    stepper = odex.ExtrapolationStepper(steppers, steps, weights, system, u0, num_cores=num_cores)

    # Solve the system, profiling
    start = time.time()
    un = stepper.step(u0, t0, dt, n, dense_output=None, observer=observer)
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
    estimated_duration = n*np.sum(np.array(steps)+1)*mean_eval_time
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
    if plot_only:
        num_cores = [4]
    else:
        num_cores = range(1,5)

    durations = [run_odex_convection_2d(nc, plot_only) for nc in num_cores]
    print('')
    for ii in range(len(num_cores)):
        print('Solver Duration ({} Thread{}): {}'.format(num_cores[ii], ' ' if num_cores[ii] == 1 else 's', durations[ii]))
    print('')


def profile_odex_convection_2d(run):
    cpus = [1, 2, 4]
    calls = []
    filenames = []
    for cpu in cpus:
        calls.append('run_odex_convection_2d({},False)'.format(cpu))
        filenames.append('conv2d_{}cpu.stats'.format(cpu))

    if run:
        for call, filename in zip(calls, filenames):
            cProfile.run(call, filename)

    for filename in filenames:
        p = pstats.Stats(filename)
        p.strip_dirs().sort_stats('cumtime').print_stats()


def sanity_check():
    for num_cores in range(1,10):
        run_odex_simple(num_cores)


def main():
    sanity_check()
#    test_odex_convection()
    test_odex_convection_2d()

    # 2D convection
#    run_odex_convection_2d(4, False, outfile='convection2d.mp4')
#    profile_odex_convection_2d(True)


if __name__=='__main__':
    main()


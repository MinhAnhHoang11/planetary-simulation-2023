## COMP1730/6730 Project assignment

# Your ANU ID: u7494091
# Your NAME: Minh Anh Hoang
# I declare that this submission is my own work
# [ref: https://www.anu.edu.au/students/academic-skills/academic-integrity ]

## You should implement the following functions
## You can define new function(s) if it helps you decompose the problem
## into smaller subproblems.

import numpy as np
import math

### Task 1 ###

def load_data(path_to_file):
    """
    Loads data from a tsv file.

    Parameters
    ----------
    path_to_file : str
        stores the current configuration of a planetary system. This includes
        the star and planets, their masses, positions and velocities. 

    Returns
    -------
    numpy.ndarray
        the array configuration of a planetary system.

    """
    return np.genfromtxt(path_to_file, delimiter='\t',
                         dtype=None, encoding=None, skip_header=1)


def count_objects(path_to_file):
    """
    Counts the number of star/planets in a planetary system. 

    Parameters
    ----------
    path_to_file : str
        stores the current configuration of a planetary system. This includes
        the star and planets, their masses, positions and velocities.

    Returns
    -------
    int
        the number of star/planets in a planetary system.

    """
    table = load_data(path_to_file)
    return table.shape[0]


def compute_position(path_to_file, time):
    """
    Predicts the configuration at a "future" time point t1 (in seconds). 

    Parameters
    ----------
    path_to_file : str
        stores the current configuration of a planetary system. This includes
        the star and planets, their masses, positions and velocities.
    time : non-negative float
        the time (in seconds). 

    Returns
    -------
    p_new : list of lists
        position of each object at the new time point. 

    """
    table = load_data(path_to_file)
    number_of_objects = count_objects(path_to_file)
    
    # Create a table with [number_of_objects] rows and 2 columns. 
    p_new = []
    for i in range(number_of_objects):
        x = table[i][2] + table[i][4] * time
        y = table[i][3] + table[i][5] * time
        p_new.append([x, y])
    
    return p_new


### Task 2 ###

def calculate_distance(path_to_file, time):
    """
    Calculates the Euclidean distance between two objects in a planetary system. 

    Parameters
    ----------
    path_to_file : str
        stores the current configuration of a planetary system. This includes
        the star and planets, their masses, positions and velocities.
    time : non-negative float
        the time (in seconds).

    Returns
    -------
    distance : list
        stores the distance between any two objects in a planetary system.

    """
    number_of_objects = count_objects(path_to_file)
    
    position_t1 = compute_position(path_to_file, time)
    
    distance = []
    for i in range(number_of_objects):
        for j in range(number_of_objects):
            x_i = position_t1[i][0]
            x_j = position_t1[j][0]
                
            y_i = position_t1[i][1]
            y_j = position_t1[j][1]
                
            dist = math.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
            
            distance.append(dist)
    
    return distance


def compute_force(path_to_file, time):
    """
    Computes the gravitational force between two objects depending on their 
    masses and the distance between them. 

    Parameters
    ----------
    path_to_file : str
        stores the current configuration of a planetary system. This includes
        the star and planets, their masses, positions and velocities.
    time : non-negative float
        the time (in seconds).

    Returns
    -------
    force : list of lists
        the gravitational force between two objects.

    """
    table = load_data(path_to_file)
    number_of_objects = count_objects(path_to_file)

    position_t1 = compute_position(path_to_file, time)
    distance = calculate_distance(path_to_file, time)
    mass = [ table[i][1] for i in range(number_of_objects) ]
    
    # Create a table with [number_of_objects - 1] rows and 2 columns.
    force = []
    for i in range(0, number_of_objects):
        for j in range(0, number_of_objects):
            # Exclude the object itself. 
            if i != j:
                G = 6.67e-11
                
                f_x = \
                    ((G*mass[i]*mass[j]) / distance[i*number_of_objects+j]**2) * ((position_t1[j][0] - position_t1[i][0]) / distance[i*number_of_objects+j])
    
                f_y = \
                    ((G*mass[i]*mass[j]) / distance[i*number_of_objects+j]**2) * ((position_t1[j][1] - position_t1[i][1]) / distance[i*number_of_objects+j])
                
                force.append([f_x, f_y])
    
    return force


def compute_acceleration(path_to_file, time):
    """
    Computes the acceleration of each object at the time point t1 as given by
    Newton's second law of motion. 

    Parameters
    ----------
    path_to_file : str
        stores the current configuration of a planetary system. This includes
        the star and planets, their masses, positions and velocities.
    time : non-negative float
        the time (in seconds).

    Returns
    -------
    acceleration : list of lists
        the acceleration of each object. 

    """
    table = load_data(path_to_file)
    number_of_objects = count_objects(path_to_file)
    
    mass = [ table[i][1] for i in range(number_of_objects) ]
    force = compute_force(path_to_file, time)
    
    # Create a table with [number_of_objects] row and 2 columns.
    acceleration = []
    for i in range(number_of_objects):
        F_slice = \
            force[i*(number_of_objects-1):(i*(number_of_objects-1)+(number_of_objects-1))]
        
        F_x = sum([ F_slice[row][0] for row in range(len(F_slice)) ])
        F_y = sum([ F_slice[row][1] for row in range(len(F_slice)) ])
        
        a_x = F_x / mass[i]
        a_y = F_y / mass[i]
        
        acceleration.append([a_x, a_y])
    
    return acceleration


### Task 3 ###

def point_position(path_to_file, dt, current_position, previous_velocity):
    """
    Computes a new position based on the current position and previous velocity. 

    Parameters
    ----------
    path_to_file : str
        stores the current configuration of a planetary system. This includes
        the star and planets, their masses, positions and velocities.
    dt : float
        the difference in time (in seconds) 
    current_position : list
        the positions of the objects at the current time point. 
    previous_velocity : list of lists
        the velocity at previous time point. 

    Returns
    -------
    point_position : list
        the new positions of the objects.

    """
    number_of_objects = count_objects(path_to_file)
    
    point_position = []
    for j in range(number_of_objects):
        px_new = current_position[j*2] + previous_velocity[j][0] * dt
        py_new = current_position[j*2+1] + previous_velocity[j][1] * dt
        point_position.extend([px_new, py_new])
    
    return point_position


def distance_on_position(path_to_file, current_position):
    """
    Calculates the Euclidean distance between two objects in a planetary system,
    based on their current positions. 

    Parameters
    ----------
    path_to_file : str
        stores the current configuration of a planetary system. This includes
        the star and planets, their masses, positions and velocities.
    current_position : list
        the positions of the objects at the current time point. 

    Returns
    -------
    distance : list
        the distance between any two objects. 

    """
    number_of_objects = count_objects(path_to_file)
    
    distance = []
    for i in range(number_of_objects):
        for j in range(number_of_objects):
            x_i = current_position[i*2]
            x_j = current_position[j*2]
                
            y_i = current_position[i*2+1]
            y_j = current_position[j*2+1]
                
            dist = math.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
            
            distance.append(dist)
            
    return distance


def force_on_position(path_to_file, current_position):
    """
    Computes the gravitational force between two objects, based on their 
    current positions. 

    Parameters
    ----------
    path_to_file : str
        stores the current configuration of a planetary system. This includes
        the star and planets, their masses, positions and velocities.
    current_position : list
        the positions of the objects at the current time point. 

    Returns
    -------
    f : list of list
        the gravitational force between two objects.

    """
    table = load_data(path_to_file)
    number_of_objects = count_objects(path_to_file)
    mass = [ table[i][1] for i in range(number_of_objects) ]
    
    distance = distance_on_position(path_to_file, current_position)
    
    # Create a table with [number_of_objects - 1] rows and 2 columns.
    f = []
    for i in range(number_of_objects):
        for j in range(number_of_objects):
            # Exclude the object itself
            if i != j:
                G = 6.67e-11
                
                f_x = \
                    ((G*mass[i]*mass[j]) / distance[i*number_of_objects+j]**2) * ((current_position[j*2] - current_position[i*2]) / distance[i*number_of_objects+j])
    
                f_y = \
                    ((G*mass[i]*mass[j]) / distance[i*number_of_objects+j]**2) * ((current_position[j*2+1] - current_position[i*2+1]) / distance[i*number_of_objects+j])
                
                f.append([f_x, f_y])
    
    return f


def acceleration_on_position(path_to_file, current_position):
    """
    Computes the acceleration of the objects at a time point, based on their 
    current positions. 

    Parameters
    ----------
    path_to_file : str
        stores the current configuration of a planetary system. This includes
        the star and planets, their masses, positions and velocities.
    current_position : list
        the positions of the objects at the current time point. 
        Assumption: there is no missing values. 

    Returns
    -------
    point_acceleration : list of lists 
        the acceleration of each object.

    """
    table = load_data(path_to_file)
    number_of_objects = count_objects(path_to_file)
    mass = [ table[i][1] for i in range(number_of_objects) ]
    
    f = force_on_position(path_to_file, current_position)
    
    # Create a table with [number_of_objects] row and 2 columns.
    point_acceleration = []
    for i in range(number_of_objects):
        F_slice = f[i*(number_of_objects-1):(i*(number_of_objects-1)+(number_of_objects-1))]
        F_x = sum([ F_slice[row][0] for row in range(len(F_slice)) ])
        F_y = sum([ F_slice[row][1] for row in range(len(F_slice)) ])
        
        a_x = F_x / mass[i]
        a_y = F_y / mass[i]
        
        point_acceleration.append([a_x, a_y])
    
    return point_acceleration


def current_velocity(path_to_file, previous_velocity, current_acceleration, dt):
    """
    Computes a new velocity based on the previous velocity and 
    current acceleration.

    Parameters
    ----------
    path_to_file : str
        stores the current configuration of a planetary system. This includes
        the star and planets, their masses, positions and velocities.
    previous_velocity : list of lists
        the velocity at previous time point. 
    current_acceleration : list of lists
        he acceleration of each object.
    dt : float
        the difference in time (in seconds)

    Returns
    -------
    point_velocity : list of lists
        the velocity at current time point. 

    """
    number_of_objects = count_objects(path_to_file)
    
    point_velocity = []
    for i in range(number_of_objects):
        vx_new = previous_velocity[i][0] + current_acceleration[i][0] * dt
        vy_new = previous_velocity[i][1] + current_acceleration[i][1] * dt
        point_velocity.append([vx_new, vy_new])
    
    return point_velocity


def forward_simulation(path_to_file, total_time, num_steps):  
    """
    Predicts the positions of all objects at a series of "future" time points. 

    Parameters
    ----------
    path_to_file : str
        stores the current configuration of a planetary system. This includes
        the star and planets, their masses, positions and velocities.
    total_time : non-negative float
        time interval in seconds.
    num_steps : non-negative int
        number of time steps. 

    Returns
    -------
    result : list of lists
        the positions of all objects at the time points. 

    """
    dt = total_time / num_steps
    
    
    table = load_data(path_to_file)
    number_of_objects = count_objects(path_to_file)
    velocity = [ [table[i][4], table[i][5]] for i in range(0, number_of_objects) ]
    
    
    result = []
    
    
    # Row 0 will be the new position at time point t1 (t=total_time/num_step). 
    position_list = compute_position(path_to_file, dt)
    position_t1 = []
    for i in range(len(position_list)):
        position_t1.extend(position_list[i])
    result.append(position_t1)
    
    
    if num_steps > 1:
        # Step 1: calculate the acceleration at the current time point.
        acceleration_t1 = compute_acceleration(path_to_file, dt)
        
        
        # Step 2: calculate a new velocity.
        velocity_t1 = []
        for i in range(number_of_objects):
            vx_new = velocity[i][0] + acceleration_t1[i][0] * dt
            vy_new = velocity[i][1] + acceleration_t1[i][1] * dt
            velocity_t1.append([vx_new, vy_new])
        
        
        # Step 3: update the position, velocity and acceleration. 
        current_position = position_t1
        previous_velocity = velocity_t1
        current_acceleration = acceleration_t1
        
        
        # Repeat the Steps 1-3 for all time points going forward.
        for time in range(1, num_steps):
            # update position. 
            current_position = point_position(path_to_file, dt, current_position,
                                              previous_velocity)
            result.append(current_position)
            
            
            # update acceleration. 
            current_acceleration = acceleration_on_position(path_to_file, 
                                                            current_position)
            
            
            # update velocity.
            previous_velocity = current_velocity(path_to_file, previous_velocity, 
                                                 current_acceleration, dt)
            
            
    return result


### Task 4 ###

def max_accurate_time(path_to_file, epsilon): 
    """
    Computes the largest time interval such that the error in the position 
    predictions is smaller than epsilon.

    Parameters
    ----------
    path_to_file : str
        stores the current configuration of a planetary system. This includes
        the star and planets, their masses, positions and velocities.
    epsilon : float
        measured in metres. 

    Returns
    -------
    int
        the maximum time (in seconds). 

    """
    # Note: this is an incorrect attempt at implementation. 
    number_of_objects = count_objects(path_to_file)
    
    
    for T in range(1, 10000):
        good_config = forward_simulation(path_to_file, T, 10)
        good_position = np.array(good_config[-1]).reshape((-1,2))
        approx_config = compute_position(path_to_file, T)
        
        error = 0
        for i in range(number_of_objects):
            for j in range(number_of_objects):
                x_i = good_position[i][0]
                x_j = approx_config[j][0]
                    
                y_i = good_position[i][1]
                y_j = approx_config[j][1]
                    
                dist = ((x_i - x_j)**2 + (y_i - y_j)**2)**0.5
                error += dist
        

        if error < epsilon:
            continue
        else:
            break
        

    return T - 1


### THIS FUNCTION IS ONLY FOR COMP6730 STUDENTS ###
def orbit_time(path_to_file, object_name):
    pass

################################################################################
#                  VISUALISATION
################################################################################

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
import numpy as np

def plot_configuration(xpositions, ypositions, object_names = []):
    """
        Plot the planetary system
        xpositions: sequence of x-coordinates of all objects
        ypositions: sequence of y-coordinates of all objects
        object_names: (optional) names of the objects
    """

    fig, ax = plt.subplots()
    
    marker_list = ['o', 'X', 's']
    color_list = ['r', 'b', 'y', 'm']
    if len(object_names) == 0:
        object_names = list(range(len(xpositions)))

    for i, label in enumerate(object_names):
          ax.scatter(xpositions[i], 
                     ypositions[i], 
                     c=color_list[i%len(color_list)],
                     marker=marker_list[i%len(marker_list)], 
                     label=object_names[i],
                     s=70)
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.grid()

    plt.xlabel("x-coordinate (meters)")
    plt.ylabel("y-coordinate (meters)")
    plt.tight_layout()
    plt.show()

def visualize_forward_simulation(table, num_steps_to_plot, object_names = []):
    """
        visualize the results from forward_simulation
        table: returned value from calling forward_simulation
        num_steps_to_plot: number of selected rows from table to plot
        object_names: (optional) names of the objects
    """
    table = np.array(table)
    if len(object_names) == 0:
        object_names = list(range(len(table[0])//2))

    assert len(object_names)==len(table[0])//2

    fig = plt.figure()
    num_objects = len(table[0])//2
    xmin = min(table[:,0])
    xmax = max(table[:,0])
    ymin = min(table[:,1])
    ymax = max(table[:,1])
    for i in range(1, num_objects):
        xmin = min(xmin, min(table[:,i*2]))
        xmax = max(xmax, max(table[:,i*2]))
        ymin = min(ymin, min(table[:,(i*2+1)]))
        ymax = max(ymax, max(table[:,(i*2+1)]))

    ax = plt.axes(xlim=(xmin, 1.2*xmax), ylim=(ymin, 1.2*ymax))

    k=len(table[0])//2

    lines=[]
    for j in range(1,k): # Here we are assuming that the first object is the star
       line, = ax.plot([], [], lw=2, label=object_names[j])
       line.set_data([], [])
       lines.append(line)

    N=len(table)
    def animate(i):
        print(i)
        step_increment=N//num_steps_to_plot
        for j in range(1,k): # Here we are assuming that the first object is the star
           leading_object_trajectories=table[0:i*step_increment]
           x = [ ts[2*j] for ts in leading_object_trajectories ]
           y = [ ts[2*j+1] for ts in leading_object_trajectories ]
           lines[j-1].set_data(x, y)
        return lines
    
    fig.legend()
    plt.grid()
    matplotlib.rcParams['animation.embed_limit'] = 1024
    anim = FuncAnimation(fig, animate, frames=num_steps_to_plot, interval=20, blit=False)
    plt.show()
    return anim

## Un-comment the lines below to show an animation of 
## the planets in the solar system during the next 100 years, 
## using 10000 time steps, with only 200 equispaced time steps 
## out of these 10000 steps in the whole [0,T] time interval 
## actually plotted on the animation
#object_trajectories=forward_simulation("solar_system.tsv", 31536000.0*100, 10000)
#animation=visualize_forward_simulation(object_trajectories, 200)

## Un-comment the lines below to show an animation of 
## the planets in the TRAPPIST-1 system during the next 20 DAYS, 
## using 10000 time steps, with only 200 equispaced time steps 
## out of these 10000 steps in the whole [0,T] time interval 
## actually plotted on the animation
#object_trajectories=forward_simulation("trappist-1.tsv", 86400.0*20, 10000)
#animation=visualize_forward_simulation(object_trajectories, 200)

################################################################################
#               DO NOT MODIFY ANYTHING BELOW THIS POINT
################################################################################    

def test_compute_position():
    '''
    Run tests of the forward_simulation function.
    If all tests pass you will just see "all tests passed".
    If any test fails there will be an error message.
    NOTE: passing all tests does not automatically mean that your code is correct
    because this function only tests a limited number of test cases.
    '''
    position = np.array(compute_position("solar_system.tsv", 86400))
    truth = np.array([[ 0.00000000e+00,  0.00000000e+00],
       [ 4.26808000e+10,  2.39780800e+10],
       [-1.01015040e+11, -3.75684800e+10],
       [-1.13358400e+11, -9.94612800e+10],
       [ 3.08513600e+10, -2.11534304e+11],
       [ 2.11071360e+11, -7.44638848e+11],
       [ 6.54704160e+11, -1.34963798e+12],
       [ 2.37964662e+12,  1.76044582e+12],
       [ 4.39009072e+12, -8.94536896e+11]])
    assert len(position) == len(truth)
    for i in range(0, len(truth)):
        assert len(position[i]) == len(truth[i])
        if np.linalg.norm(truth[i]) == 0.0:
            assert np.linalg.norm(position[i] - truth[i]) < 1e-6
        else:    
            assert np.linalg.norm(position[i] - truth[i])/np.linalg.norm(truth[i]) < 1e-6
    print("all tests passed")
    
def test_compute_acceleration():
    '''
    Run tests of the compute_acceleration function.
    If all tests pass you will just see "all tests passed".
    If any test fails there will be an error message.
    NOTE: passing all tests does not automatically mean that your code is correct
    because this function only tests a limited number of test cases.
    '''
    acceleration = np.array(compute_acceleration("solar_system.tsv", 10000))
    truth = np.array([[ 3.42201832e-08, -2.36034356e-07],
       [-4.98530431e-02, -2.26599078e-02],
       [ 1.08133552e-02,  3.71768441e-03],
       [ 4.44649916e-03,  3.78461924e-03],
       [-3.92422837e-04,  2.87361538e-03],
       [-6.01036812e-05,  2.13176213e-04],
       [-2.58529454e-05,  5.32663462e-05],
       [-1.21886258e-05, -9.01929841e-06],
       [-6.48945783e-06,  1.32120968e-06]])
    assert len(acceleration) == len(truth)
    for i in range(0, len(truth)):
        assert len(acceleration[i]) == len(truth[i])
        assert np.linalg.norm(acceleration[i] - truth[i])/np.linalg.norm(truth[i]) < 1e-6
    print("all tests passed")

def test_forward_simulation():
    '''
    Run tests of the forward_simulation function.
    If all tests pass you will just see "all tests passed".
    If any test fails there will be an error message.
    NOTE: passing all tests does not automatically mean that your code is correct
    because this function only tests a limited number of test cases.
    '''
    trajectories = forward_simulation("solar_system.tsv", 31536000.0*100, 10)

    known_position = np.array([[ 1.63009260e+10,  4.93545018e+09],
       [-8.79733713e+13,  1.48391575e+14],
       [ 3.54181417e+13, -1.03443654e+14],
       [ 5.85930535e+13, -7.01963073e+13],
       [ 7.59849728e+13,  1.62880599e+13],
       [ 2.89839690e+13,  1.05111979e+13],
       [ 3.94485026e+12,  6.29896920e+12],
       [ 2.84544375e+12, -3.06657485e+11],
       [-4.35962396e+12,  2.04187940e+12]])
   
    rtol = 1.0e-6
    last_position = np.array(trajectories[-1]).reshape((-1,2))
    for j in range(len(last_position)):
        x=last_position[j]
        assert np.linalg.norm(x-known_position[j])/np.linalg.norm(known_position[j]) < rtol, "Test Failed!"

    print("all tests passed")

def test_max_accurate_time():
    '''
    Run tests of the max_accurate_time function.
    If all tests pass you will just see "all tests passed".
    If any test fails there will be an error message.
    NOTE: passing all tests does not automatically mean that your code is correct
    because this function only tests a limited number of test cases.
    '''
    assert max_accurate_time('solar_system.tsv', 1) == 5.0
    assert max_accurate_time('solar_system.tsv', 1000) == 163.0
    assert max_accurate_time('solar_system.tsv', 100000) == 1632.0
    print("all tests passed")

def test_orbit_time():
    '''
    Run tests of the orbit_time function.
    If all tests pass you will just see "all tests passed".
    If any test fails there will be an error message.
    NOTE: passing all tests does not automatically mean that your code is correct
    because this function only tests a limited number of test cases.
    '''

    # accepting error of up to 10%
    assert abs(orbit_time('solar_system.tsv', 'Mercury')/7211935.0 - 1.0) < 0.1
    assert abs(orbit_time('solar_system.tsv', 'Venus')/19287953.0 - 1.0) < 0.1
    assert abs(orbit_time('solar_system.tsv', 'Earth')/31697469.0 - 1.0) < 0.1
    assert abs(orbit_time('solar_system.tsv', 'Mars')/57832248.0 - 1.0) < 0.1
    print("all tests passed")

# Formation control
# Authors:
# Mohamed Aboraya
# Marco Ghaly

from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
import os
import numpy as np
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import networkx as nx

#####################################
# Functions
#####################################

# system dynamics



MAXITERS = 2000
visu_frequency = 0.5 #[Hz]
COMM_TIME = 1.0/visu_frequency

n_x = 2     # dimension of x_i
DELTA = 1.0/visu_frequency
def square_shape():
    NN = 4      # number of agents
    L = 2.0
    D = (L**2+L**2)**0.5
    distances = [[0, L, D, L],
                 [L, 0, L, D],
                 [D, L, 0, L],
                 [L, D, L, 0]]
    distances = np.asarray(distances)
    Adj = np.float32(distances>0)
    return distances, Adj, NN

def polygon_shape():
    NN = 6      # number of agents
    L = 2
    D = 2*L
    H = np.sqrt(3)*L

    # minimally rigid 2*N-3 (only for regular polygons)
    # rigid
    distances = [[0,     L,      0,    D,     H,    L],
                [L,     0,      L,    0,     D,    0],
                [0,     L,      0,    L,     0,    D],     
                [D,     0,      L,    0,     L,    0],     
                [H,     D,      0,    L,     0,    L],     
                [L,     0,      D,    0,     L,    0]]
    distances = np.asarray(distances)
    Adj = np.float32(distances>0)
    return distances, Adj, NN
def A_shape():
    NN = 6      # number of agents
    L = 2
    ang = np.deg2rad(120)
    D = ((L/2)**2 + L**2 - 2*(L/2)*L*np.cos(ang))**0.5
    S = (L**2 - (L/2)**2)**0.5
    DL = 2*L
    # LD = (L**2+L**2-2*L*L*np.cos(ang))**0.5
    LD = 0
    distances = [[ 0,      L,    2*L,      0,     LD,     DL],
                [  L,      0,      L,    L/2,     L,     LD],
                [2*L,      L,      0,      S,     L,    2*L],     
                [  0,    L/2,      S,      0,   L/2,      0],     
                [  LD,      L,      L,    L/2,     0,      L],     
                [ DL,     LD,    2*L,      0,     L,     0]]
    distances = np.asarray(distances)
    Adj = np.float32(distances>0)
    return distances, Adj, NN

distances, Adj, NN = polygon_shape()

# definite initial positions
x_init = np.random.rand(n_x*NN)*5
# x_init = np.array([-0.7313, 0.7819, -0.0173, -0.6565, 1.4212, 0.0575, 0.701, 1.4959])

# x_init = [-10.0, 10.0, -20.0, 20.0]

def generate_launch_description():
    launch_description = [] # Append here your nodes
    ld = LaunchDescription()




    ################################################################################
    # RVIZ
    ################################################################################

    # initialize launch description with rviz executable
    rviz_config_dir = get_package_share_directory('consensus')
    rviz_config_file = os.path.join(rviz_config_dir, 'rviz_config.rviz')

    # launch_description.append(
    node_rviz = Node(
        package='rviz2', 
        executable='rviz2', 
        arguments=['-d', rviz_config_file],
        # output='screen',
        # prefix='xterm -title "rviz2" -hold -e'
        )
    ld.add_action(node_rviz)

    ################################################################################

    for ii in range(NN):
        N_ii = np.nonzero(Adj[:, ii])[0].tolist()
        # N_i = np.where(Adj[:,i]>0)[0]
        Adj_ii = Adj[:,ii].tolist()
        index_ii =  ii*n_x + np.arange(n_x)
        x_init_ii = x_init[index_ii].tolist()
        distances_ii = distances[ii,:].tolist()


        ld.add_action(
            Node(
                package='formation_control',
                namespace =f'agent_{ii}',
                executable='the_agent',
                parameters=[{ # dictionary
                                'agent_id': ii, 
                                'neigh': N_ii, 
                                'x_init': x_init_ii,
                                'max_iters': MAXITERS,
                                'Adj_ii': Adj_ii,
                                'comm_time': COMM_TIME,
                                'dist_ii': distances_ii
                                }],
                output='screen',
                prefix=f'xterm -title "agent_{ii}" -hold -e',
            ))
        ld.add_action(
            Node(
                package='formation_control', 
                namespace='agent_{}'.format(ii),
                executable='visualizer', 
                parameters=[{
                                'agent_id': ii,
                                'node_frequency': visu_frequency,
                                }],
            ))

    return ld#LaunchDescription(launch_description)
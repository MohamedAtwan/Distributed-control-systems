# Formation control
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



MAXITERS = 3000
visu_frequency = 0.5 #[Hz]
COMM_TIME = 1.0/visu_frequency

n_x = 2     # dimension of x_i
DELTA = 1.0/visu_frequency
n_leaders = 1
def square_shape():
    NN = 4      # number of agents
    L = 1.0
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
x_init = np.random.rand(n_x*NN)
# x_init = np.array([-0.7313, 0.7819, -0.0173, -0.6565, 1.4212, 0.0575, 0.701, 1.4959])
# x_init = np.array([0.2230, 2.3727, 2.0696, 1.5883, 2.3044, -0.4000, 0.7038, -1.6035, -1.1375, -0.8162, -1.3795, 1.1717])
# x_init = [-10.0, 10.0, -20.0, 20.0]
I_nx = np.eye(n_x)

DEGREE = np.sum(Adj,axis=0) 
D_IN = np.diag(DEGREE)
L_IN = D_IN - Adj.T

L_f = L_IN[0:NN-n_leaders,0:NN-n_leaders]
L_fl = L_IN[0:NN-n_leaders,NN-n_leaders:]

# followers dynamics
LL = np.concatenate((L_f, L_fl), axis = 1)

# leaders dynamics
LL = np.concatenate((LL, np.zeros((n_leaders,NN))), axis = 0)

# replicate for each dimension -> kronecker product
LL_kron = np.kron(LL,I_nx)

A = - LL_kron

BB_kron = np.zeros((NN*n_x,n_leaders*n_x))
BB_kron[(NN-n_leaders)*n_x:,:] = np.identity(n_x*n_leaders, dtype=int)

B = BB_kron
def generate_launch_description():
    launch_description = [] # Append here your nodes
    ld = LaunchDescription()




    ################################################################################
    # RVIZ
    ################################################################################

    # initialize launch description with rviz executable
    rviz_config_dir = get_package_share_directory('formation_control')
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
        A_ii = A[index_ii,:].flatten().tolist()
        B_ii = B[index_ii,:].flatten().tolist()

        if ii >= (NN-n_leaders): # if agent is a leader
            agent_type = "leader"

        else:
            agent_type = "follower"


        ld.add_action(
            Node(
                package='formation_control',
                namespace =f'agent_{ii}',
                executable='the_agent_drive',
                parameters=[{ # dictionary
                                'agent_id': ii,
                                'agent_type': agent_type,
                                'num_leaders': n_leaders,
                                'neigh': N_ii,
                                'A_ii':A_ii,
                                'B_ii':B_ii,
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
                executable='visualizer_labelled', 
                parameters=[{
                                'agent_id': ii,
                                'agent_type':agent_type,
                                'node_frequency': visu_frequency,
                                }],
            ))

    return ld#LaunchDescription(launch_description)
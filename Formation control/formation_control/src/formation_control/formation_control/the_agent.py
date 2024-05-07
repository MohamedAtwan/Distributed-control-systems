#
# Formation Control Algorithm
# Mohamed Aboraya, Marco Ghaly, Domenico petrella
# Bologna, 20/06/2023

from time import sleep
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
import warnings


def form_func(idx,x, dist, Adj,N_ii, n_x, delta = 2.0):
    xdot = x.copy()
    NN = Adj.shape[0]
    index_ii =  idx*n_x + np.arange(n_x)
    for jj in N_ii:
        index_jj = jj*n_x + np.arange(n_x)
        xx_ii = x[index_ii]
        xx_jj = x[index_jj]
        barrier = (xx_ii-xx_jj)/np.dot((xx_ii-xx_jj),(xx_ii-xx_jj))
        dV_ij = 20*(np.dot((xx_ii-xx_jj),(xx_ii-xx_jj)) - dist[jj]**2)*(xx_ii - xx_jj)- 5*barrier#(1/(xx_ii-xx_jj))
        xdot[index_ii] = xdot[index_ii] - delta* dV_ij
    return xdot

class Agent(Node):
    def __init__(self):
        super().__init__('agent',
                            allow_undeclared_parameters=True,
                            automatically_declare_parameters_from_overrides=True)
            
        # Get parameters from launch file
        self.agent_id = self.get_parameter('agent_id').value
        # self.neigh = self.get_parameter('neigh').value
        # 
        self.x_i = np.array(self.get_parameter('x_init').value)
        self.n_x = self.x_i.shape[0]
        self.kk = 0
        self.max_iters = self.get_parameter('max_iters').value
        

        # Adjacency matrix
        self.Adj = np.array(self.get_parameter('Adj_ii').value)
        self.neigh = self.get_parameter('neigh').value
        print(self.neigh)
        self.degree = len(self.neigh)
        self.distances = np.array(self.get_parameter('dist_ii').value)
        self.N_agents = len(self.distances)

        # definite initial positions
        self.x_i = np.random.rand(self.n_x)#+1)*np.array([np.random.choice([-1,1]) for _ in range(self.n_x)])

        # create a subscription to each neighbor
        qos = QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT
        for j in self.neigh:
            self.create_subscription(
                                    MsgFloat, 
                                    f'/topic_{j}', #  topic_name
                                    self.listener_callback,qos)
                                    # qos_profile=rclpy.qos.qos_profile_sensor_data)
        
        # create the publisher
        qos = QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT
        self.publisher = self.create_publisher(
                                            MsgFloat, 
                                            f'/topic_{self.agent_id}',
                                            qos)
        self.timer_period = self.get_parameter('comm_time').value # [seconds]
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        # initialize a dictionary with a list of received messages from each neighbor j [a queue]
        self.received_data = { j:[] for j in self.neigh }

        print(f"Setup of agent {self.agent_id} completed")

    def listener_callback(self, msg):
        # self.get_logger().info(f'msg data: {list(msg.data[1:])}')
        self.received_data[int(msg.data[0])].append(list(msg.data[1:]))
        # return None

    def timer_callback(self):
        # Skip the first iteration
        if self.kk > 0: 
            # Have all messages at time kk-1 been received? Check top message in each queue
            # if self.agent_id == 0:
            #     print([self.received_data[j][0][0] for j in self.neigh])
            all_received = all([(self.kk-1) == self.received_data[j][0][0] if self.received_data[j] else False for j in self.neigh]) # True if all True
            if all_received:
                x = []
                for cnt in range(self.N_agents):
                    if cnt == self.agent_id: #not (cnt in list(sellf.received_data.keys()):
                        x+= [self.x_i[0], self.x_i[1]]
                    elif not(cnt in self.neigh):
                        x+= [np.nan, np.nan]
                    else:
                        tmp = self.received_data[cnt].pop(0)
                        x += tmp[-2:]

                x = np.array(x)
                with np.testing.suppress_warnings() as sup:
                    sup.filter(DeprecationWarning)
                    x_dot = form_func(self.agent_id, x, self.distances, self.Adj, self.neigh, self.n_x, 1e-3)
                index_ii =  self.agent_id*self.n_x + np.arange(self.n_x)
                self.x_i = x_dot[index_ii]
                
                # Stop the node if kk exceeds the maximum iteration
                if self.kk > self.max_iters:
                    print("\nMAXITERS reached")
                    sleep(3) # [seconds]
                    self.destroy_node()

        # Publish the updated message
        msg = MsgFloat()
        msg.data = [float(self.agent_id), float(self.kk), self.x_i[0], self.x_i[1]]
        self.publisher.publish(msg)

        self.get_logger().info(f"Agent {int(msg.data[0]):d} -- Iter = {int(msg.data[1]):d} \t coords = ({msg.data[2]:.4f}, {msg.data[3]:.4f})")
        # update iteration counter
        self.kk += 1

def main():
    warnings.filterwarnings('ignore')
    rclpy.init()

    agent = Agent()
    agent.get_logger().info(f"Agent {agent.agent_id:d} -- Waiting for sync...")
    sleep(1)
    agent.get_logger().info("GO!")

    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        agent.get_logger().info("----- Node stopped cleanly -----")
    finally:
        rclpy.shutdown() 

if __name__ == '__main__':
    main()
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32MultiArray as msg_float

class Visualizer(Node):

    def __init__(self):
        super().__init__('visualizer',
                            allow_undeclared_parameters=True,
                            automatically_declare_parameters_from_overrides=True)
        
        # Get parameters from launcher
        self.agent_id = self.get_parameter('agent_id').value
        self.node_frequency = self.get_parameter('node_frequency').value
        self.agent_type = self.get_parameter('agent_type').value


        #######################################################################################
        # Let's subscribe to the topic we want to visualize
        
        self.subscription = self.create_subscription(
                                                     msg_float, 
                                                     '/topic_{}'.format(self.agent_id),
                                                     self.listener_callback, 
                                                     10)

        #######################################################################################

        # Create the publisher that will communicate with Rviz
        self.timer = self.create_timer(1.0/self.node_frequency, self.publish_data)
        self.publisher = self.create_publisher(
                                                Marker, 
                                                '/visualization_topic', 
                                                1)

        # Initialize the current_pose method (in this example you can also use list or np.array)                                         
        self.current_pose = Pose()

        self.color = [1.0, 0.0, 0.0, 1.0]
        if self.agent_type == "leader":
            self.color = [0.0, 1.0, 0.0, 1.0]


    def listener_callback(self, msg):
        # store (and rearrange) the received message
        self.current_pose.position.x = msg.data[2]#0.2*self.agent_id # fix x coordinate
        self.current_pose.position.y = msg.data[3]
            
    def publish_data(self):
        if self.current_pose.position is not None:
            # Set the type of message to send to Rviz -> Marker
            # (see http://docs.ros.org/en/noetic/api/visualization_msgs/html/index-msg.html)
            marker = Marker()

            # Select the name of the reference frame, without it markers will be not visualized
            marker.header.frame_id = 'my_frame'
            marker.header.stamp = self.get_clock().now().to_msg()

            # Select the type of marker
            marker.type = Marker.SPHERE

            # set the pose of the marker (orientation is omitted in this example)
            marker.pose.position.x = self.current_pose.position.x
            marker.pose.position.y = self.current_pose.position.y
            marker.pose.position.z = self.current_pose.position.z 

            # Select the marker action (ADD, DELATE)
            marker.action = Marker.ADD

            # Select the namespace of the marker
            marker.ns = 'agents'

            # Let the marker be unique by setting its id
            marker.id = self.agent_id

            # Specify the scale of the marker
            scale = 0.2
            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale

            # Specify the color of the marker as RGBA
            # color = [1.0, 0.0, 0.0, 1.0]
            # # if self.agent_type == "leader":
            # if self.agent_id == 3:
            #     color = [0.0, 0.5, 0.5, 1.0]

            marker.color.r = self.color[0]
            marker.color.g = self.color[1]
            marker.color.b = self.color[2]
            marker.color.a = self.color[3]

            # Let's publish the marker
            self.publisher.publish(marker)

def main():
    rclpy.init()

    visualizer = Visualizer()

    try:
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        print("----- Visualizer stopped cleanly -----")
    finally:
        rclpy.shutdown() 

if __name__ == '__main__':
    main()
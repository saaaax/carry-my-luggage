import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
import rospy
from geometry_msgs.msg import PoseStamped
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
class KalmanFilter3D:
    def __init__(self):
        """Inicializa el filtro de Kalman para estimar la posición y velocidad en 3D."""
        # Crear el filtro de Kalman
        self.kf = KalmanFilter(dim_x=6, dim_z=3)  # Estado [x, y, z, vx, vy, vz], medición [x, y, z]

        # Estado inicial [x, y, z, vx, vy, vz]
        self.kf.x = np.array([[0.],    # Posición x inicial
                              [0.],    # Posición y inicial
                              [0.],    # Posición z inicial
                              [0.],    # Velocidad en x inicial
                              [0.],    # Velocidad en y inicial
                              [0.]])   # Velocidad en z inicial

        # Matriz de transición del estado (supone movimiento constante)
        self.kf.F = np.array([[1., 0., 0., 1., 0., 0.],  # x = x + vx * dt
                              [0., 1., 0., 0., 1., 0.],  # y = y + vy * dt
                              [0., 0., 1., 0., 0., 1.],  # z = z + vz * dt
                              [0., 0., 0., 1., 0., 0.],  # vx = vx
                              [0., 0., 0., 0., 1., 0.],  # vy = vy
                              [0., 0., 0., 0., 0., 1.]]) # vz = vz

        # Matriz de medición (solo medimos la posición x, y, z)
        self.kf.H = np.array([[1., 0., 0., 0., 0., 0.],  # Solo observamos x
                              [0., 1., 0., 0., 0., 0.],  # Solo observamos y
                              [0., 0., 1., 0., 0., 0.]]) # Solo observamos z

        # Matriz de covarianza del proceso
        self.kf.P *= 1000.  # Alta incertidumbre inicial

        # Ruido del proceso
        self.kf.Q = np.eye(6) * 0.1  # Ajustamos el ruido del proceso (pequeño)

        # Ruido de medición (covarianza del ruido en las mediciones de posición)
        self.kf.R = np.array([[5., 0., 0.], 
                              [0., 5., 0.], 
                              [0., 0., 5.]])  # Ruido en x, y, z

    def predict_and_update(self, measurement):
        """
        Realiza la predicción y actualización del filtro de Kalman usando una medición.
        
        Args:
            measurement (array): Una medición de la posición [x, y, z].
        
        Returns:
            np.array: La estimación actualizada de la posición [x, y, z].
        """
        # Realizar la predicción
        self.kf.predict()

        # Actualizar con la medición
        self.kf.update(measurement.reshape(3, 1))

        # Devolver la estimación de la posición (x, y, z)
        return self.kf.x[:3].reshape(3)

class PoseReader:

    def __init__(self):
        self.kf2d = KalmanFilter2D()
        self.estimates = []
        self.sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.callback_cam)
        self.yolo_pub = rospy.Publisher('yolo_pub', Image, queue_size=10)
        self.bridge = CvBridge()
        self._image_data = None
        self.last_estimated_position = None
        self.position=[]

        # Suscribirse al tema '/person_pose'
        self.sub = rospy.Subscriber("/person_pose", PoseStamped, self.callback)

    def callback(self, msg):
        # Extraer las coordenadas x, y de PoseStamped
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        
        self.position=[x,y,z]
        # Crear la medición como un array
        measurement = np.array([x, y, z])
        
        # Aplicar el filtro de Kalman
        self.last_estimated_position = self.kf2d.predict_and_update(measurement)
        self.estimates.append(self.last_estimated_position)

        rospy.loginfo(f"Medición: ({x}, {y}, {z}), Estimación: {self.last_estimated_position}")

    def callback_cam(self, msg):
        # Convertir el mensaje de imagen de ROS a OpenCV
        self.last_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        x, y, z = self.position

        if self.last_estimated_position is not None:
            
            image = cv2.circle(self.last_image, (int(x), int(y), int(z)), radius=3, color=(0, 255, 0), thickness=-1)
            image_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
            self.yolo_pub.publish(image_msg)

        

if __name__ == '__main__':
    rospy.init_node('kalman', anonymous=True)
    pose_reader = PoseReader()
    
    # Mantener el nodo vivo
    rospy.spin()
    
    

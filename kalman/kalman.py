import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
import rospy
from geometry_msgs.msg import PoseStamped
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
class KalmanFilter2D:
    def __init__(self):
        """Inicializa el filtro de Kalman para estimar la posición y velocidad en 2D."""
        # Crear el filtro de Kalman
        self.kf = KalmanFilter(dim_x=4, dim_z=2)  # Estado [x, y, vx, vy], medición [x, y]

        # Estado inicial [x, y, vx, vy]
        self.kf.x = np.array([[0.],    # Posición x inicial
                              [0.],    # Posición y inicial
                              [0.],    # Velocidad en x inicial
                              [0.]])   # Velocidad en y inicial

        # Matriz de transición del estado (supone movimiento constante)
        self.kf.F = np.array([[1., 0., 1., 0.],  # x = x + vx * dt
                              [0., 1., 0., 1.],  # y = y + vy * dt
                              [0., 0., 1., 0.],  # vx = vx
                              [0., 0., 0., 1.]]) # vy = vy

        # Matriz de medición (solo medimos la posición x e y)
        self.kf.H = np.array([[1., 0., 0., 0.],  # Solo observamos x
                              [0., 1., 0., 0.]]) # Solo observamos y

        # Matriz de covarianza del proceso
        self.kf.P *= 1000.  # Alta incertidumbre inicial

        # Ruido del proceso
        self.kf.Q = np.eye(4) * 0.1  # Ajustamos el ruido del proceso (pequeño)

        # Ruido de medición (covarianza del ruido en las mediciones de posición)
        self.kf.R = np.array([[5., 0.], 
                              [0., 5.]])  # Ruido en x e y

    def predict_and_update(self, measurement):
        """
        Realiza la predicción y actualización del filtro de Kalman usando una medición.
        
        Args:
            measurement (array): Una medición de la posición [x, y].
        
        Returns:
            np.array: La estimación actualizada de la posición [x, y].
        """
        # Realizar la predicción
        self.kf.predict()

        # Actualizar con la medición
        self.kf.update(measurement.reshape(2, 1))

        # Devolver la estimación de la posición (x, y)
        return self.kf.x[:2].reshape(2)

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
        
        self.position=[x,y]
        # Crear la medición como un array
        measurement = np.array([x, y])
        
        # Aplicar el filtro de Kalman
        self.last_estimated_position = self.kf2d.predict_and_update(measurement)
        self.estimates.append(self.last_estimated_position)

        rospy.loginfo(f"Medición: ({x}, {y}), Estimación: {self.last_estimated_position}")

    def callback_cam(self, msg):
        # Convertir el mensaje de imagen de ROS a OpenCV
        self.last_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        x, y = self.position

        if self.last_estimated_position is not None:
            
            image = cv2.circle(self.last_image, (int(x), int(y)), radius=3, color=(0, 255, 0), thickness=-1)
            image_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
            self.yolo_pub.publish(image_msg)

        

if __name__ == '__main__':
    rospy.init_node('kalman', anonymous=True)
    pose_reader = PoseReader()
    
    # Mantener el nodo vivo
    rospy.spin()
    
    

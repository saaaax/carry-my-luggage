# Std Libs
import numpy as np
import math
np.float = np.float64  # Deprecado en versiones modernas; considerar revisión
import cv2
from cv_bridge import CvBridge, CvBridgeError

# ROS Python Libs
import rospy
import ros_numpy as rnp

# ROS msgs
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray

# YOLO
from ultralytics import YOLO


class PersonLocator:
    def __init__(self):
        self.model = YOLO('yolov8n-pose.pt')
        rospy.loginfo('YOLOv8 is now running...')
        
        # Suscriptores
        self.points_sub= rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.callback_points)
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback_image)
        
        # Publicadores
        self.yolo_pub = rospy.Publisher('/yolo_pub', Image, queue_size=10)
        self.pose_pub = rospy.Publisher("/person_pose", PoseStamped, queue_size=2)
        
        # Almacenamiento de datos
        self._image_data = None
        self._points_data = None

        self.xy = None
        self.bridge = CvBridge()

    def callback_image(self, msg):
        self._image_data= rnp.numpify(msg)
        print('enter')

        # Realizar detecciones con YOLO
        detections = self.model(self._image_data, show=False, conf=0.8)
        keypoints=detections[0].keypoints.xy

        # Verificar detecciones y extraer puntos clave
        if detections and len(detections[0].keypoints) > 0:
            try:
                # Extraer la posición del primer keypoint detectado
                person_x = int(keypoints[0][0][0])  # Coordenada X del keypoint
                person_y = int(keypoints[0][0][1])  # Coordenada Y del keypoint

                self.xy=[person_x, person_y]

                # Publicar el mensaje PoseStamped
                rospy.loginfo(f"Posición de la persona: x={person_x}, y={person_y}")
            except: pass

        else:
            rospy.logwarn("No se detectaron personas.")
    def callback_points(self, msg):

        try:
            #Convertir el mensaje de ROS a una imagen de profundidad en formato OpenCV
            self._points_data = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            #print("")
        except CvBridgeError as e:
            rospy.logerr(f"Error al convertir la imagen de profundidad: {e}")

        
        if self._points_data is None:
            rospy.logwarn("No se ha recibido imagen de profundidad aún.")
            return
        try: 
            x=self.xy[0]
            y=self.xy[1]

            z = self._points_data[x,y]

            #Crear un mensaje PoseStamped
            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = "camera_frame"

            #Asignar posición
            pose_msg.pose.position.x = float(x)
            pose_msg.pose.position.y = float(y)
            pose_msg.pose.position.z= float(z)

            self.pose_pub.publish(pose_msg)


            rospy.loginfo(f"Posición de la persona: x={x}, y={y}, z={z}")

        except:
            print("Aun no se recibe x,y")

        

if __name__ == '__main__':
    rospy.init_node('person_pose', anonymous=True)
    person_locator = PersonLocator()
    rospy.spin()

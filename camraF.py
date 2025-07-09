import cv2
import cv2.aruco as aruco
import numpy as np

# === Cargar parámetros de la cámara desde archivo ===
npzfile = np.load("/home/proaxeltv/Desktop/PROYECTO ROBOTICA/parametros_calibracion4.npz")
camera_matrix = npzfile["mtx"]
dist_coeffs = npzfile["dist"]

# Diccionario de nombres por ID
id_to_name = {
    0: "Robot 1",
    1: "Robot 2",
    2: "Robot 3",
    4: "Caja 1",
    5: "Caja 2",
    6: "Caja 3"
}

# Orden fijo para imprimir
orden_ids = [0, 1, 2, 4, 5, 6]

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# === Iniciar cámara ===
cap = cv2.VideoCapture(0)  # Usa la cámara por defecto

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

mode = None
marker_size = 50  # mm

last_poses = {}
posicion_umbral = 5.0  # mm
rvec_dict = {}
tvec_dict = {}

def print_positions(current_mode, poses):
    print("\033c", end="")
    if current_mode == 'cam':
        print("Líneas desde la cámara a los objetos:")
        for id in orden_ids:
            if id in poses:
                pos = poses[id]
                name = id_to_name.get(id, f"ID {id}")
                distancia = np.linalg.norm(pos)
                print(f"{name}: Posición (x, y, z) = {pos}, Distancia = {distancia:.2f} mm")
    elif current_mode in ('robot1', 'robot2', 'robot3'):
        robot_id = {'robot1': 0, 'robot2': 1, 'robot3': 2}[current_mode]
        if robot_id not in poses:
            print(f"No se detectó {id_to_name.get(robot_id, f'Robot {robot_id}')}")
            return

        R_robot, _ = cv2.Rodrigues(rvec_dict[robot_id])
        t_robot = tvec_dict[robot_id].reshape(3, 1)
        T_robot = np.vstack((np.hstack((R_robot, t_robot)), [0, 0, 0, 1]))
        T_robot_inv = np.linalg.inv(T_robot)

        print(f"Distancias desde {id_to_name.get(robot_id)} (referencial local):")
        for id in orden_ids:
            if id in poses and id != robot_id:
                R_obj, _ = cv2.Rodrigues(rvec_dict[id])
                t_obj = tvec_dict[id].reshape(3, 1)
                T_obj = np.vstack((np.hstack((R_obj, t_obj)), [0, 0, 0, 1]))

                T_local = T_robot_inv @ T_obj
                delta = T_local[:3, 3]
                distancia = np.linalg.norm(delta)

                name = id_to_name.get(id, f"ID {id}")
                print(f"{name} respecto a {id_to_name.get(robot_id)}: Δx={delta[0]:.2f}, Δy={delta[1]:.2f}, Δz={delta[2]:.2f}, Distancia = {distancia:.2f} mm")
    else:
        print("\033c", end="")

def draw_lines_from_marker(frame, centers, marker_id, color=(0, 255, 255)):
    if marker_id not in centers:
        return
    start = centers[marker_id]
    for other_id, other_center in centers.items():
        if other_id != marker_id:
            cv2.line(frame, start, other_center, color, 2)

def posiciones_diferentes(p1, p2, umbral=5.0):
    if p1.keys() != p2.keys():
        return True
    for key in p1:
        if np.linalg.norm(p1[key] - p2[key]) > umbral:
            return True
    return False

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer de la cámara.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    poses = {}
    centers = {}

    if ids is not None:
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

        for i in range(len(ids)):
            marker_id = int(ids[i][0])
            poses[marker_id] = tvecs[i][0]
            rvec_dict[marker_id] = rvecs[i][0]
            tvec_dict[marker_id] = tvecs[i][0]
            c = corners[i][0]
            center_x = int(np.clip(c[:, 0].mean(), 0, frame.shape[1] - 1))
            center_y = int(np.clip(c[:, 1].mean(), 0, frame.shape[0] - 1))
            centers[marker_id] = (center_x, center_y)

        aruco.drawDetectedMarkers(frame, corners, ids)
        for i in range(len(ids)):
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_size / 2)

        for marker_id in poses:
            name = id_to_name.get(marker_id, f"ID {marker_id}")
            center = centers[marker_id]
            cv2.putText(frame, name, (center[0], center[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        h, w = frame.shape[:2]
        center_x_cam = w // 2
        center_y_cam = h // 2
        cv2.line(frame, (center_x_cam - 10, center_y_cam), (center_x_cam + 10, center_y_cam), (0, 0, 255), 2)
        cv2.line(frame, (center_x_cam, center_y_cam - 10), (center_x_cam, center_y_cam + 10), (0, 0, 255), 2)

        if mode == 'cam':
            for center in centers.values():
                cv2.line(frame, (center_x_cam, center_y_cam), center, (255, 0, 0), 2)
        elif mode == 'robot1':
            draw_lines_from_marker(frame, centers, 0)
        elif mode == 'robot2':
            draw_lines_from_marker(frame, centers, 1)
        elif mode == 'robot3':
            draw_lines_from_marker(frame, centers, 2)

    if mode is not None:
        if posiciones_diferentes(poses, last_poses, posicion_umbral):
            print_positions(mode, poses)
            last_poses = poses.copy()

    cv2.imshow("Marcadores detectados", frame)

    key = cv2.waitKey(30) & 0xFF
    if key == 27 or key == ord('q'):
        break
    elif key == ord('p'):
        mode = None if mode == 'cam' else 'cam'
        print_positions(mode, poses)
        last_poses = poses.copy()
    elif key == ord('o'):
        mode = None if mode == 'robot1' else 'robot1'
        print_positions(mode, poses)
        last_poses = poses.copy()
    elif key == ord('i'):
        mode = None if mode == 'robot2' else 'robot2'
        print_positions(mode, poses)
        last_poses = poses.copy()
    elif key == ord('u'):
        mode = None if mode == 'robot3' else 'robot3'
        print_positions(mode, poses)
        last_poses = poses.copy()

cap.release()
cv2.destroyAllWindows()

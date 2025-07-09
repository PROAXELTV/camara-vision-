import cv2
import cv2.aruco as aruco
import numpy as np

# Parámetros de cámara simulada
fx, fy = 800, 800
cx, cy = 640 / 2, 480 / 2
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

# Diccionario de nombres por ID
id_to_name = {
    0: "Robot 1",
    1: "Robot 2",
    2: "Robot 3",
    4: "Caja 1",
    5: "Caja 2",
    6: "Caja 3"
}

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()

# Captura desde la cámara real
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("No se pudo acceder a la cámara.")
    exit()

marker_size = 22  # en mm
mode = None  # Modo actual ('cam', 'robot1', 'robot2', 'robot3', None)

poses = {}
centers = {}

def print_positions(current_mode):
    print("\033c", end="")  # limpiar consola
    if current_mode == 'cam':
        print("Líneas desde la cámara a los objetos:")
        for id, pos in poses.items():
            name = id_to_name.get(id, f"ID {id}")
            distancia = np.linalg.norm(pos)
            print(f"{name}: Posición (x, y, z) = {pos}, Distancia = {distancia:.2f} mm")
    elif current_mode in ('robot1', 'robot2', 'robot3'):
        robot_id = {'robot1': 0, 'robot2': 1, 'robot3': 2}[current_mode]
        if robot_id not in poses:
            print(f"No se detectó {id_to_name.get(robot_id, f'Robot {robot_id}')}")
            return
        pos_robot = poses[robot_id]
        name_robot = id_to_name.get(robot_id, f"Robot {robot_id}")
        print(f"Distancias desde {name_robot}:")
        for id, pos in poses.items():
            if id == robot_id:
                continue
            name = id_to_name.get(id, f"ID {id}")
            delta = pos - pos_robot
            dist = np.linalg.norm(delta)
            print(f"{name} respecto a {name_robot}: Δx={delta[0]:.2f}, Δy={delta[1]:.2f}, Δz={delta[2]:.2f}, Distancia = {dist:.2f} mm")
    else:
        print("\033c", end="")

def draw_lines_from_marker(frame, marker_id, color=(0, 255, 255)):
    if marker_id not in centers:
        return
    start = centers[marker_id]
    for other_id, other_center in centers.items():
        if other_id != marker_id:
            cv2.line(frame, start, other_center, color, 2)
    cv2.line(frame, start, (int(cx), int(cy)), color, 2)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el frame de la cámara.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    poses.clear()
    centers.clear()

    if ids is not None:
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

        for i in range(len(ids)):
            id = int(ids[i][0])
            poses[id] = tvecs[i][0]
            c = corners[i][0]
            center_x = int(np.clip(c[:, 0].mean(), 0, frame.shape[1] - 1))
            center_y = int(np.clip(c[:, 1].mean(), 0, frame.shape[0] - 1))
            centers[id] = (center_x, center_y)

        aruco.drawDetectedMarkers(frame, corners, ids)
        for i in range(len(ids)):
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_size / 2)

        for id in poses.keys():
            name = id_to_name.get(id, f"ID {id}")
            pos = poses[id]
            center = centers[id]
            distancia = np.linalg.norm(pos)
            cv2.putText(frame, name, (center[0], center[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            texto_dist = f"{distancia:.1f} mm"
            cv2.putText(frame, texto_dist, (center[0], center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Dibujar ejes centrales de cámara
    size = 10
    color_red = (0, 0, 255)
    center_x_cam = int(cx)
    center_y_cam = int(cy)
    cv2.line(frame, (center_x_cam - size, center_y_cam), (center_x_cam + size, center_y_cam), color_red, 2)
    cv2.line(frame, (center_x_cam, center_y_cam - size), (center_x_cam, center_y_cam + size), color_red, 2)

    # Dibujar líneas según el modo
    if mode == 'cam':
        for center in centers.values():
            cv2.line(frame, (center_x_cam, center_y_cam), center, (255, 0, 0), 2)
    elif mode == 'robot1':
        draw_lines_from_marker(frame, 0)
    elif mode == 'robot2':
        draw_lines_from_marker(frame, 1)
    elif mode == 'robot3':
        draw_lines_from_marker(frame, 2)

    cv2.imshow("Marcadores detectados", frame)

    key = cv2.waitKey(30) & 0xFF
    if key == 27 or key == ord('q'):
        break
    elif key == ord('p'):
        mode = None if mode == 'cam' else 'cam'
        print_positions(mode)
    elif key == ord('o'):
        mode = None if mode == 'robot1' else 'robot1'
        print_positions(mode)
    elif key == ord('i'):
        mode = None if mode == 'robot2' else 'robot2'
        print_positions(mode)
    elif key == ord('u'):
        mode = None if mode == 'robot3' else 'robot3'
        print_positions(mode)

cap.release()
cv2.destroyAllWindows()

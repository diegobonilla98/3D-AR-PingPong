import open3d
import time
import numpy as np
import cv2


cam = cv2.VideoCapture(2)

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
parameters = cv2.aruco.DetectorParameters_create()

ball_position = np.array([0., 0., 0.], np.float32)
ball_velocity = np.array([0., 0., 0.], np.float32)
ball_acceleration = np.array([0., -0.007, 0.5], np.float32)
ball_radius = 1.

frame_count = 0

plane_y = -20
plane_width = 50

pala_w, pala_h = 5, 10
pala_adv_pos = np.array([0., 0., - plane_width])
pala_adv_vel = np.array([0., 0., 0.])
difficulty = 0.07

pala_my_pos = np.array([0., 0., plane_width])

net_height = 5.

friction = 0.9
strength = 1.


def animation(visualizer):
    global ball_position, ball_velocity, ball_acceleration, frame_count, pala_adv_vel, pala_adv_pos, pala_my_pos
    frame_count += 1

    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    if ids is not None:
        if ids[0][0] == 2:
            coords = corners[0][0]
            cv2.polylines(frame, [np.int32(coords)], True, (0, 255, 0))
            pos = np.int32(np.mean(coords, axis=0))
            pala_my_pos[0] = (((pos[0] / (frame.shape[1] / 2)) * 2) - 1) * (plane_width / 2)
            pala_my_pos[1] = ((frame.shape[0] - pos[1]) / (frame.shape[0] / 2) * 70) - 50

    cv2.imshow("Result", frame)
    cv2.waitKey(1)

    visualizer.clear_geometries()

    plane = open3d.geometry.TriangleMesh.create_box(width=plane_width, height=0.5, depth=plane_width * 2)
    plane.paint_uniform_color([0.07, 0.36, 0.78])
    plane.compute_vertex_normals()
    plane.translate((-plane_width / 2, plane_y - 1, -plane_width))

    net = open3d.geometry.TriangleMesh.create_box(width=plane_width, height=net_height, depth=0.5)
    net.paint_uniform_color([0.8, 0.8, 0.8])
    net.compute_vertex_normals()
    net.translate((-plane_width / 2, plane_y, 0.))

    pala_adv = open3d.geometry.TriangleMesh.create_box(width=pala_w, height=pala_h, depth=0.5)
    pala_adv.paint_uniform_color([1., 0., 0.])
    pala_adv.compute_vertex_normals()
    pala_adv.translate(pala_adv_pos)

    pala_my = open3d.geometry.TriangleMesh.create_box(width=pala_w, height=pala_h, depth=0.5)
    pala_my.paint_uniform_color([1., 0., 0.])
    pala_my.compute_vertex_normals()
    pala_my.translate(pala_my_pos)

    sphere = open3d.geometry.TriangleMesh.create_sphere(radius=ball_radius, resolution=20)
    sphere.paint_uniform_color([1., 0.5, 0.])
    sphere.compute_vertex_normals()

    sphere.translate(ball_position)

    visualizer.add_geometry(sphere)
    visualizer.add_geometry(plane)
    visualizer.add_geometry(net)
    visualizer.add_geometry(pala_adv)
    visualizer.add_geometry(pala_my)

    ctr = visualizer.get_view_control()
    ctr.rotate(50, 60, 0)
    # ctr.rotate(0, 500, 0)
    # ctr.rotate(500, 0, 0)

    if ball_position[1] - ball_radius + ball_velocity[1] < plane_y:
        ball_velocity[1] = - ball_velocity[1] * friction

    if ball_position[2] + ball_radius < - plane_width:
        if pala_adv_pos[0] < ball_position[0] < pala_adv_pos[0] + pala_w and pala_adv_pos[1] < ball_position[1] < pala_adv_pos[1] + pala_h:
            ball_velocity[2] = - ball_velocity[2] * np.random.uniform(1., 4.)
            ball_velocity[1] = abs(ball_velocity[1])
            ball_velocity[1] += np.random.uniform(0.2, 0.3)
            ball_velocity[0] = np.random.uniform(-0.1, 0.1)
        else:
            ball_position = np.array([0., 0., 0.], np.float32)
            ball_velocity = np.array([0., 0., 0.], np.float32)
            ball_acceleration = np.array([0., -0.007, 0.5], np.float32)

    if ball_position[2] + ball_radius > plane_width:
        if pala_my_pos[0] < ball_position[0] < pala_my_pos[0] + pala_w and pala_my_pos[1] < ball_position[1] < pala_my_pos[1] + pala_h:
            ball_velocity[2] = - ball_velocity[2] * 1.3
            ball_velocity[1] = abs(ball_velocity[1])
            ball_velocity[1] += 0.3
            ball_velocity[0] = - ball_position[0] * 0.001
        else:
            ball_position = np.array([0., 0., 0.], np.float32)
            ball_velocity = np.array([0., 0., 0.], np.float32)
            ball_acceleration = np.array([0., -0.007, -0.5], np.float32)

    if ball_position[2] < 0 and ball_velocity[2] < 0:
        if abs(ball_velocity[2]) > 0.3:
            ball_velocity[2] -= ball_velocity[2] * 0.05
        magnet_vector = ball_position - (pala_adv_pos + [pala_w / 2, pala_h, 0])
        pala_adv_vel[:2] = magnet_vector[:2] * difficulty

    if abs(ball_velocity[2]) > 0.3 and ball_position[2] > 0 and ball_velocity[2] > 0:
        pala_adv_vel = np.array([0., 0., 0.])
        ball_velocity[2] -= ball_velocity[2] * 0.05

    if ball_position[1] > plane_width / 5:
        ball_acceleration[1] -= 0.006

    pala_adv_pos += pala_adv_vel
    pala_adv_pos[1] = np.clip(pala_adv_pos[1], plane_y, 50)

    ball_velocity += ball_acceleration
    ball_velocity = np.clip(ball_velocity, -1, 1)
    ball_position += ball_velocity
    ball_acceleration = np.array([0., -0.007, 0.], np.float32)


sphere = open3d.geometry.TriangleMesh.create_sphere(radius=1., resolution=20)
open3d.visualization.draw_geometries_with_animation_callback([], animation, window_name="Ping3DPong", left=2000)

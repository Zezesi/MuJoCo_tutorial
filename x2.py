# mjpython x2.py
import time
import numpy as np
import mujoco
import mujoco.viewer
from simple_pid import PID


def quaternion_to_euler(q_w, q_x, q_y, q_z):
    roll = np.atan2(2 * (q_w * q_x + q_y * q_z), 1 - 2 * (q_x ** 2 + q_y ** 2))  # roll
    pitch = np.asin(2 * (q_w * q_y - q_z * q_x))  # pitch
    yaw = np.atan2(2 * (q_w * q_z + q_x * q_y), 1 - 2 * (q_y ** 2 + q_z ** 2))  # yaw
    return roll, pitch, yaw


class PIDController:
    def __init__(self, targetPos, vel_limit=2.5) -> None:
        self.targetPos = targetPos  # desired position
        self.vel_limit = vel_limit
        self.pid_x = PID(1.9, 0.14, 1.6, setpoint=self.targetPos[0],
                         output_limits=(-vel_limit, vel_limit), )
        self.pid_y = PID(1.9, 0.14, 1.6, setpoint=self.targetPos[1],
                         output_limits=(-vel_limit, vel_limit))
        self.pid_z = PID(0.551, 0.058, 0.12, setpoint=targetPos[2],
                         output_limits=(-vel_limit, vel_limit))

    def __call__(self, curPos: np.array):
        desVel = np.array([0, 0, 0])
        # curPos is the current position
        desVel[0] = self.pid_x(curPos[0])  # desired velocity(action) in x direction in world frame
        desVel[1] = self.pid_y(curPos[1])  # desired velocity(action) in y direction in world frame
        desVel[2] = self.pid_z(curPos[2])  # desired velocity(action) in z direction in world frame
        return desVel

    def update_targetPos(self, targetPos):  # update the desired position in the next step

        self.targetPos = targetPos
        self.pid_x.setpoint = self.targetPos[0]
        self.pid_y.setpoint = self.targetPos[1]
        self.pid_z.setpoint = self.targetPos[2]


class IMUSensor:
    def __init__(self, data):
        self.position = data.qpos  # 7 numbers (3D position(x,y,z) followed by unit quaternion(1,i,j,k)) in world frame
        self.velocity = data.qvel  # 6 numbers (3D linear velocity(x,y,z) in world frame followed by 3D angular velocity in body frame).
        self.acceleration = data.qacc  # same as the velocity

    def get_position(self):
        return self.position

    def get_velocity(self):
        return self.velocity

    def get_acceleration(self):
        return self.acceleration


class quadcopter:
    def __init__(self, targetPos=np.array((0, 0, 0))):
        self.model = mujoco.MjModel.from_xml_path('skydio_x2/scene.xml')
        self.data = mujoco.MjData(self.model)

        self.controller = PIDController(targetPos=targetPos)
        self.sensor = IMUSensor(self.data)

        # angular displacement control to stabalize inflight dynamics
        self.pid_roll = PID(2.68, 0.56, 1.25, setpoint=0, output_limits=(-1, 1))
        self.pid_pitch = PID(2.68, 0.56, 1.25, setpoint=0, output_limits=(-1, 1))
        self.pid_yaw = PID(0.54, 0.001, 5.36, setpoint=0, output_limits=(-np.pi, np.pi))

        # velocity control to reach the desired position
        self.pid_v_x = PID(0.11, 0.002, 0.03, setpoint=0,
                           output_limits=(-0.1, 0.1))
        self.pid_v_y = PID(0.11, 0.002, 0.03, setpoint=0,
                           output_limits=(-0.1, 0.1))
        self.pid_v_z = PID(0.11, 0.002, 0.03, setpoint=0,
                           output_limits=(-0.1, 0.1))

    def update_angle_conrol(self):
        curVel = self.sensor.get_velocity()  # current velocity(linear in world frame, angular in body frame)
        curPos = self.sensor.get_position()[:3]  # current linear position(x,y,z) in world frame

        desVel = self.controller(curPos=curPos)  # desired velocity in world frame

        self.pid_v_x.setpoint = desVel[0]  # update reference velocity in x direction in world frame
        self.pid_v_y.setpoint = desVel[1]  # update reference velocity in y direction in world frame
        self.pid_v_z.setpoint = desVel[2]  # update reference velocity in z direction in world frame

        desPitch = self.pid_v_x(
            curVel[0])  # desired change of velocity in x direction is achieved by the change of the pitch angle
        desRoll = -self.pid_v_y(
            curVel[1])  # desired change of velocity in y direction is achieved by the change of the roll angle

        self.pid_roll.setpoint = desRoll  # update reference roll position in world frame
        self.pid_pitch.setpoint = desPitch  # update reference pitch position in world frame

    def update_motor_control(self):
        curVel = self.sensor.get_velocity()
        curAng = self.sensor.get_position()[3:]  # current angular position(quaternion(1,i,j,k)
        curRoll, curPitch, curYaw = quaternion_to_euler(curAng[0], curAng[1], curAng[2], curAng[3])

        cmd_thrust = self.pid_v_z(curVel[
                                      2]) + 3.2495  # desired change of velocity in z direction is achieved by the change of the thrust force
        cmd_roll = self.pid_roll(curRoll)  # desired roll torque
        cmd_pitch = self.pid_pitch(curPitch)  # desired pitch torque
        cmd_yaw = self.pid_yaw(curYaw)  # desired yaw torque

        # motor control
        out = self.compute_motor_control(cmd_thrust, cmd_roll, cmd_pitch, cmd_yaw)
        self.data.ctrl[:4] = out

    #  as the drone is underactuated we set
    def compute_motor_control(self, thrust, roll, pitch, yaw):
        motor_control = [
            thrust - roll + pitch - yaw,
            thrust + roll + pitch + yaw,
            thrust + roll - pitch - yaw,
            thrust - roll - pitch + yaw,
        ]
        return motor_control


if __name__ == "__main__":
    quad = quadcopter(targetPos=np.array([0, 0, 1]))  # target is the desired position in world(fixed) frame

    with mujoco.viewer.launch_passive(quad.model, quad.data) as viewer:
        viewer.cam.fixedcamid = mujoco.mj_name2id(quad.model, mujoco.mjtObj.mjOBJ_CAMERA, "track")
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        time.sleep(2)  # wait 2 seconds
        start = time.time()
        step = 1

        while viewer.is_running() and time.time() - start < 80:
            step_start = time.time()

            # flight program
            if time.time() - start > 3:
                quad.controller.update_targetPos(np.array([0, 0, 2]))

            if time.time() - start > 12:
                quad.controller.update_targetPos(np.array([10, 10, 5]))

            if time.time() - start > 24:
                quad.controller.update_targetPos(np.array([-20, -10, 3]))

            quad.update_angle_conrol()
            quad.update_motor_control()

            mujoco.mj_step(quad.model, quad.data)

            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(quad.data.time % 2)

            viewer.sync()

            step += 1

            time_until_next_step = quad.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

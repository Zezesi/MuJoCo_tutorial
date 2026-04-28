# mjpython cf2.py
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
    def __init__(self, targetPos) -> None:
        self.targetPos = targetPos  # desired position

        self.pid_thrust = PID(0.11, 0.002, 0.3, setpoint=targetPos[2],output_limits=(-0.26487, 0.08513))
        self.pid_roll = PID(-0.11, -0.002, -0.3, setpoint=targetPos[1], output_limits=(-0.01, 0.01))
        self.pid_pitch = PID(0.11, 0.002, 0.3, setpoint=targetPos[0], output_limits=(-0.01, 0.01))
        self.pid_yaw = PID(0.11, 0.002, 0.3, setpoint=0.0, output_limits=(-np.pi, np.pi))


    def update_ctrl(self, curPos: np.array,curYaw):

        thrust = self.pid_thrust(curPos[2])+0.26487 # desired thrust
        roll = self.pid_roll(curPos[1])  # desired roll angle
        pitch = self.pid_pitch(curPos[0])  # desired pitch angle
        yaw = self.pid_yaw(curYaw)  # desired yaw angle
        return thrust,roll,pitch,yaw

    def update_targetPos(self, targetPos):  # update the desired position for the next step

        self.targetPos = targetPos
        self.pid_thrust.setpoint=self.targetPos[2]
        self.pid_roll.setpoint = self.targetPos[1]
        self.pid_pitch.setpoint = self.targetPos[0]


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
    def __init__(self, targetPos=np.array((0.0, 0.0, 0.0))):
        self.model = mujoco.MjModel.from_xml_path('bitcraze_crazyflie_2/scene.xml')
        self.data = mujoco.MjData(self.model)

        self.controller = PIDController(targetPos=targetPos)
        self.sensor = IMUSensor(self.data)

        self.pid_roll_m = PID(0.11, 0.002, 0.3, setpoint=0, output_limits=(-1.0, 1.0))
        self.pid_pitch_m = PID(0.11, 0.002, 0.3, setpoint=0, output_limits=(-1.0, 1.0))
        self.pid_yaw_m = PID(0.11, 0.002, 0.3, setpoint=0, output_limits=(-1.0, 1.0))


    def update_motor_control(self):
        # curVel = self.sensor.get_velocity()
        curPos = self.sensor.get_position()[:3]  # current linear position(x,y,z) in world frame
        curAng = self.sensor.get_position()[3:]  # current angular position(quaternion(1,i,j,k)
        curRoll, curPitch, curYaw = quaternion_to_euler(curAng[0], curAng[1], curAng[2], curAng[3])
        desThrust,desRoll,desPitch,desYaw=self.controller.update_ctrl(curPos,curYaw)

        self.pid_roll_m.setpoint=desRoll
        self.pid_pitch_m.setpoint=desPitch
        self.pid_yaw_m.setpoint=desYaw

        thrust=desThrust
        rollTorque=self.pid_roll_m(curRoll) # desired roll torque
        pitchToruqe=self.pid_pitch_m(curPitch) # desired pitch torque
        yawTorque=self.pid_yaw_m(curYaw) # desired yaw torque

        self.data.ctrl[:4] = [thrust,-rollTorque,-pitchToruqe,-yawTorque]
        print(curPos)





if __name__ == "__main__":
    quad = quadcopter(targetPos=np.array([0.0, 0.0, 1.0]))  # target is the desired position in world(fixed) frame

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
                quad.controller.update_targetPos(np.array([2.0, 1.0, 2.0]))

            if time.time() - start > 12:
                quad.controller.update_targetPos(np.array([2.0, 1.0, 5.0]))

            if time.time() - start > 24:
                quad.controller.update_targetPos(np.array([2.0, 1.0, 10.0]))

            quad.update_motor_control()

            mujoco.mj_step(quad.model, quad.data)

            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(quad.data.time % 2)

            viewer.sync()

            step += 1

            time_until_next_step = quad.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

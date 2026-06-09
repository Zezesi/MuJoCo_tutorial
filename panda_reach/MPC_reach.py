import casadi as ca
import numpy as np
import time
import mujoco
import mujoco.viewer
Ts=0.05


def transformation_matrix(x,y,z,r,p,yaw):
    t=np.array([[np.cos(p)*np.cos(yaw)                                 , -np.sin(yaw)*np.cos(p)                                  , np.sin(p)           , x],
                  [np.sin(p)*np.sin(r)*np.cos(yaw)+np.sin(yaw)*np.cos(r) , -np.sin(p)*np.sin(r)*np.sin(yaw) + np.cos(r)*np.cos(yaw), -np.sin(r)*np.cos(p), y],
                  [-np.sin(p)*np.cos(r)*np.cos(yaw)+np.sin(r)*np.sin(yaw), np.sin(p)*np.sin(yaw)*np.cos(r)+np.sin(r)*np.cos(yaw)   , np.cos(p)*np.cos(r) , z],
                  [0                                                     , 0                                                       , 0                   , 1]])
    return t

def dh_transformation_matrix(alpha,a,d,theta):
    dh_t = np.array([[np.cos(theta), -np.sin(theta) , 0, a],
                  [np.sin(theta) * np.cos(alpha),np.cos(theta) * np.cos(alpha), -np.sin(alpha),-d*np.sin(alpha)],
                  [np.sin(theta) * np.sin(alpha),np.cos(theta) * np.sin(alpha), np.cos(alpha) , d*np.cos(alpha)],
                  [0, 0, 0, 1]])
    return dh_t



class panda_env:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path('franka_emika_panda/mjx_scene.xml')
        self.data = mujoco.MjData(self.model)
        self.initial_qpos=np.array([-5.94958683e-17, 5.57178318e-03, -6.85235486e-06, -6.95284621e-02, -1.61440323e-04, -7.17258051e-03, -5.46813142e-06, 6.91022958e-07, -9.37611953e-08])
    def update_attack_point(self):
        x=np.random.uniform(low=-0.2, high=0.2)
        y=np.random.uniform(low=0.3, high=0.5)
        z=np.random.uniform(low=0.2, high=0.6)
        self.attack_point=np.array([x,y,z])


    def step(self,action):
        self.data.ctrl = action
        mujoco.mj_step(self.model, self.data)
        state = np.concatenate((self.data.qpos[0:8], self.data.site("gripper").xpos), axis=0)
        return state,self.data.site("attack_point0").xpos

    def reset(self):
        self.data.qpos=self.initial_qpos
        self.update_attack_point()
        self.model.site("attack_point0").pos = self.attack_point
        mujoco.mj_step(self.model, self.data)
        time.sleep(0.1) # make sure that it has enough time to reset in mujoco
        state = np.concatenate((self.data.qpos[0:8],self.data.site("gripper").xpos),axis=0)
        return state,self.data.site("attack_point0").xpos

if __name__ == "__main__":
    N = 10  # prediction horizon length
    u0_min = -2.8973
    u0_max = 2.8973
    u1_min = -1.7628
    u1_max = 1.7628
    u2_min = -2.8973
    u2_max = 2.8973
    u3_min = -3.0718
    u3_max = -0.0698
    u4_min = -2.8973
    u4_max = 2.8973
    u5_min = -0.0175
    u5_max = 3.7525
    u6_min = -2.8973
    u6_max = 2.8973
    u7_min = 0.0
    u7_max = 0.04
    U_min=np.tile(np.array([u0_min,u1_min,u2_min,u3_min,u4_min,u5_min,u6_min,u7_min]),N).reshape(N,-1)
    U_max=np.tile(np.array([u0_max,u1_max,u2_max,u3_max,u4_max,u5_max,u6_max,u7_max]),N).reshape(N,-1)
    offset=np.array([-5.94958683e-17, 5.57178318e-03, -6.85235486e-06, -6.95284621e-02, -1.61440323e-04, -7.17258051e-03, -5.46813142e-06, 6.91022958e-07]).reshape(1,8)
    dht01 = dh_transformation_matrix(0.0, 0.0, 0.333, 0.0)
    dht12 = dh_transformation_matrix(270 / 180 * np.pi, 0.0, 0.0, 0.0)
    dht23 = dh_transformation_matrix(90 / 180 * np.pi, 0.0, 0.316, 0.0)
    dht34 = dh_transformation_matrix(90 / 180 * np.pi, 0.0825, 0.0, 175 / 180 * np.pi) # after finetuning, it is about 175 degrees not 180 degrees
    dht45 = dh_transformation_matrix(90 / 180 * np.pi, 0.0825, 0.384, 180 / 180 * np.pi)
    dht56 = dh_transformation_matrix(90 / 180 * np.pi, 0.0, 0.0, 0.0)
    dht67 = dh_transformation_matrix(90 / 180 * np.pi, 0.088, 0.207, 315 / 180 * np.pi)
    dht78 = dh_transformation_matrix(270 / 180 * np.pi, 0.0, 0.0, 180 / 180 * np.pi)

    opti = ca.Opti()

    # optimal control inputs
    opt_controls = opti.variable(N, 8)

    # states
    panda_states = opti.variable(N + 1, 11)



    # parameters
    # the actual initial states
    init_panda_states = opti.parameter(1, 11)


    # attack point position
    attack_position = opti.parameter(1, 3)

    Q = np.diag([1e1, 1e1, 1e1])
    R = np.diag([0, 0, 0, 0, 0, 0, 0, 0])
    P = np.diag([1e2, 1e2, 1e2])


    # for assigning opti.variable, must use opti.subject_to; for assigning opti.parameter, use opti.set_value,but not during the optimization process; for others use =
    opti.subject_to(panda_states[0, :] == init_panda_states)
    obj=0
    for i in range(N):
        opti.subject_to(panda_states[i + 1, 0:8] == opt_controls[i,:])
        t1 = transformation_matrix(0, 0, 0, 0, 0, opt_controls[i, 0] - offset[0,0])
        t2 = transformation_matrix(0, 0, 0, 0, 0, opt_controls[i, 1] - offset[0,1])
        t3 = transformation_matrix(0, 0, 0, 0, 0, opt_controls[i, 2] - offset[0,2])
        t4 = transformation_matrix(0, 0, 0, 0, 0, opt_controls[i, 3] - offset[0,3])
        t5 = transformation_matrix(0, 0, 0, 0, 0, opt_controls[i, 4] - offset[0,4])
        t6 = transformation_matrix(0, 0, 0, 0, 0, opt_controls[i, 5] - offset[0,5])
        t7 = transformation_matrix(0, 0, 0, 0, 0, opt_controls[i, 6] - offset[0,6])
        next_gripper_position=dht01@t1@dht12@t2@dht23@t3@dht34@t4@dht45@t5@dht56@t6@dht67@t7@dht78
        opti.subject_to(panda_states[i + 1, 8] == next_gripper_position[0,3])
        opti.subject_to(panda_states[i + 1, 9] == next_gripper_position[1,3])
        opti.subject_to(panda_states[i + 1, 10] == next_gripper_position[2,3])
        opt_states1 = attack_position[0, :] - panda_states[i+1, 8:]
        opt_states2 = offset - opt_controls[i,:]

        # remember the cost function and the hard constraints can be replaced by soft functions
        # objective function
        obj += ca.mtimes([opt_states1, Q, opt_states1.T])+ca.mtimes([opt_states2, R, opt_states2.T])
    opti.minimize(obj)

    # state and action constraints
    opti.subject_to(opti.bounded(U_min, opt_controls, U_max))


    opts_setting = {'ipopt.max_iter': 10000,
                    'ipopt.print_level': 0,
                    'print_time': 0,
                    'ipopt.acceptable_tol': 1e-8,
                    'ipopt.acceptable_obj_change_tol': 1e-6}

    opti.solver('ipopt', opts_setting)

    env = panda_env()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        time.sleep(2)  # wait 2 seconds
        panda_states_step,attack_position_step=env.reset()

        while viewer.is_running():
            step_start = time.time()

            # assign parameters
            opti.set_value(init_panda_states, panda_states_step.reshape(1,11))
            opti.set_value(attack_position, attack_position_step.reshape(1,3))

            sol = opti.solve()

            # obtain the optimized control inputs
            u_res = sol.value(opt_controls)
            panda_states_step,attack_position_step=env.step(u_res[0,:])

            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(env.data.time % 2)

            viewer.sync()



            time_until_next_step = Ts - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)





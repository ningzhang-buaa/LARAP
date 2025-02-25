import numpy as np
import robosuite as suite
from robosuite import load_controller_config

env_variant=dict(
        robot_keys=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel'],
        obj_keys=['object-state'],
        controller_type='OSC_POSITION_YAW',
        controller_config_update=dict(
            position_limits=[
                [-0.30, -0.30, 0.75],
                [0.15, 0.30, 1.15]
            ],
        ),
        env_kwargs=dict(
            ignore_done=True,
            reward_shaping=True,
            hard_reset=False,
            control_freq=10,
            camera_heights=512,
            camera_widths=512,
            table_offset=[-0.075, 0, 0.8],
            reward_scale=5.0,

            skill_config=dict(
                skills=['atomic', 'open', 'reach', 'grasp', 'push'],
                aff_penalty_fac=15.0,

                base_config=dict(
                    global_xyz_bounds=[
                        [-0.30, -0.30, 0.80],
                        [0.15, 0.30, 0.95]
                    ],
                    lift_height=0.95,
                    binary_gripper=True,

                    aff_threshold=0.06,
                    aff_type='dense',
                    aff_tanh_scaling=10.0,
                ),
                atomic_config=dict(
                    use_ori_params=True,
                ),
                reach_config=dict(
                    use_gripper_params=False,
                    local_xyz_scale=[0.0, 0.0, 0.06],
                    use_ori_params=False,
                    max_ac_calls=15,
                ),
                grasp_config=dict(
                    global_xyz_bounds=[
                        [-0.30, -0.30, 0.80],
                        [0.15, 0.30, 0.85]
                    ],
                    aff_threshold=0.03,

                    local_xyz_scale=[0.0, 0.0, 0.0],
                    use_ori_params=True,
                    max_ac_calls=20,
                    num_reach_steps=2,
                    num_grasp_steps=3,
                ),
                push_config=dict(
                    global_xyz_bounds=[
                        [-0.30, -0.30, 0.80],
                        [0.15, 0.30, 0.85]
                    ],
                    delta_xyz_scale=[0.25, 0.25, 0.05],

                    max_ac_calls=20,
                    use_ori_params=True,

                    aff_threshold=[0.12, 0.12, 0.04],
                ),
            ),
        ),
    )
controller_config = load_controller_config(default_controller=env_variant['controller_type'])
controller_config_update = env_variant.get('controller_config_update', {})
controller_config.update(controller_config_update)
env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=False,
    controller_configs=controller_config,
    **env_variant['env_kwargs']
)
#
# env = suite.make(
#     env_name="Lift",
#     robots="Panda",
#     has_renderer=True,
#     has_offscreen_renderer=False,
#     use_camera_obs=False,
# )

env.reset()

for i in range(1000):
    action = np.random.randn(5)
    obs, reward, done, info = env.step(action)
    env.render()
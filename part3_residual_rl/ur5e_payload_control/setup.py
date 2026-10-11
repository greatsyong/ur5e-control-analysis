
from setuptools import find_packages, setup

package_name = 'ur5e_payload_control'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        (
            'share/' + package_name,
            ['package.xml'],
        ),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='sooyong',
    maintainer_email='sooyong@example.com',
    description='ROS 2 control for UR5e payload transportation',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'joint_state_monitor = '
            'ur5e_payload_control.joint_state_monitor:main',
            'initial_hold = '
            'ur5e_payload_control.initial_hold:main',
            'trajectory_pd = '
            'ur5e_payload_control.trajectory_pd:main',
            'trajectory_td3 = ur5e_payload_control.trajectory_td3:main',
            'trajectory_ppo = ur5e_payload_control.trajectory_ppo:main',
            'trajectory_sac = ur5e_payload_control.trajectory_sac:main',
            'trajectory_pd_4kg = ur5e_payload_control.trajectory_pd_4kg:main',
            'trajectory_pd_reverse = ur5e_payload_control.trajectory_pd_reverse:main',
	    'trajectory_ppo_reverse = ur5e_payload_control.trajectory_ppo_reverse:main',
	    'trajectory_pd_m3_a4 = ur5e_payload_control.trajectory_pd_m3_a4:main',
	    'trajectory_ppo_m3_a4 = ur5e_payload_control.trajectory_ppo_m3_a4:main',
	    'trajectory_pd_m5_a4 = ur5e_payload_control.trajectory_pd_m5_a4:main',
	    'trajectory_ppo_m5_a4 = ur5e_payload_control.trajectory_ppo_m5_a4:main',
        ],
    },
)


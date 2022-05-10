from setuptools import setup

package_name = 'rmf_human_detector'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yadunund',
    maintainer_email='yadunund@openrobotics.org',
    description='Detect humans using depthai and publish to RMF',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'human_detector_node=rmf_human_detector.human_detector_node:main'
        ],
    },
)

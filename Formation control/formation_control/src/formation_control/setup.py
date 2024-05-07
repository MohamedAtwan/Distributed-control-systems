from setuptools import setup
from glob import glob

package_name = 'formation_control'
scripts = ['the_agent', 'the_agent_labeled', 'visualizer', 'visualizer_labelled', 'the_agent_drive']

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, glob('launch/*.launch.py')),
        ('share/' + package_name, glob('resource/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mohamed',
    maintainer_email='mohamed.aboraya@studio.unibo.it',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                            '{1} = {0}.{1}:main'.format(package_name, script) for script in scripts
        ],
    },
)

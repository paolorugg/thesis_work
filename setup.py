from setuptools import setup

package_name = 'thesis'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Paolo Ruggeri',
    maintainer_email='paolo19ruggeri@gmail.com',
    description='Thesis work',
    license='3-Clause BSD License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'posenet_node = tesi.posenet_node:main',
            'client = tesi.client_v1:main',
            'server = tesi.server_v1:main',
            'apriltag = tesi.april_tag:main',
            'features = tesi.features:main'
            ],
    },
)

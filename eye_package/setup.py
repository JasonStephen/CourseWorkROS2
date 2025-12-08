from setuptools import find_packages, setup

package_name = 'eye_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # ★★★ 关键：安装 launch 文件 ★★★
        ('share/' + package_name + '/launch', ['launch/mt3.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jason',
    maintainer_email='jason@todo.todo',
    description='Eye package for MT3',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'eye_node = eye_package.eye_node:main'
        ],
    },
)


from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=[`ui_interpretation`],
    package_dir=[``:`src`],
    requires=[`std_msgs`, `rospy`]
)

setup(**setup_args)

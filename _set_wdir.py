import os
# Set project dir as wdir
# Import this script to all desired scripts in subdirs, import sufficient to change wdir of script,
# Keep this script in project dir

# Get the path of the current script (__init__.py)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set the working directory as the project directory, which corresponds to the location of this script
os.chdir(script_dir)
from move import move
from gripper import open_gripper, close_gripper


move(0.25, -0.20, 0.0, gripper=0)
# move(0.25, 0.00, 0.12, gripper=100)   # close gripper, smoother
# close_gripper(speed=25)
# move(0.25, 0.00, 0.25)   # close gripper, smoother
# move(0.15, 0.15, 0.10, gripper=0)   # close gripper, smoother
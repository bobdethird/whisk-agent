"""
Per-object grasp configurations for the SO-101 arm.

All height offsets are *positive* values (in metres) added to the object's
detected y-coordinate.  y is the vertical axis (positive = up).

Keys per config
---------------
approach_height_offset : float
    Y distance above the object pose to hover before descending.
    Larger values give the arm more time to decelerate and align.
grasp_height_offset : float
    Y distance above the object pose at which the gripper closes.
    0.0 = close at the detected centroid; positive = close higher on object.
wrist_angle : float
    Extra wrist rotation in degrees, *added* to the detected object yaw
    before the gripper approaches.  0 = use the object's yaw unchanged.
grip_width : float
    Gripper jaw opening in metres when approaching the object.  Pre-set
    by ``dispatch_tool`` before the approach so jaws are already near the
    right width when descent begins.
object_height : float
    Vertical extent of the object above its detected centroid (metres).
    Used by the trajectory planner to raise the Y-ceiling when this
    object lies under a straight-line transit path.  0.0 = negligible.
approach_axis : str
    How the gripper approaches the object in the final step:
      "y"             — top-down descent (default; cups, bowls).
      "yaw_aligned"   — horizontal slide perpendicular to the detected yaw,
                        in the XZ plane (whisks, scoops lying on a side).
      "x" | "z"       — fixed side approach along world x or z axis.
approach_standoff : float
    For side approaches, distance in metres to stand off from the target
    before sliding in.  Ignored when approach_axis == "y".
"""

GRASP_CONFIGS: dict[str, dict] = {

    # ------------------------------------------------------------------ matcha_cup
    # A ceramic cup, ~8 cm tall, ~7 cm diameter.
    # Grip just below the rim so fingers clear the opening and the cup
    # can be lifted without catching the edge.
    "matcha_cup": {
        "approach_height_offset": 0.12,   # 12 cm hover — clears the rim comfortably
        "grasp_height_offset":    0.05,   # grip at 5 cm above centroid (below rim)
        "wrist_angle":            0.0,    # straight top-down: symmetric grip on cylinder
        "grip_width":             0.07,   # 7 cm — slightly wider than cup diameter
        "object_height":          0.08,   # top of cup ~8 cm above centroid
        "approach_axis":          "y",
        "approach_standoff":      0.0,
    },

    # ----------------------------------------------------------------- matcha_bowl
    # Wide, shallow bowl, ~5 cm tall, ~12 cm diameter.
    # Top-down pinch on the outer wall; wide grip to clear the bowl's radius.
    "matcha_bowl": {
        "approach_height_offset": 0.10,   # 10 cm — bowl is shorter, needs less clearance
        "grasp_height_offset":    0.02,   # grip near the rim edge
        "wrist_angle":            0.0,    # top-down
        "grip_width":             0.09,   # wide: fingers must reach around the bowl wall
        "object_height":          0.05,
        "approach_axis":          "y",
        "approach_standoff":      0.0,
    },

    # ---------------------------------------------------------------- matcha_whisk
    # Bamboo whisk, ~11 cm handle + 6 cm tine bundle.
    # Lies on its side; grip mid-handle with a yaw-aligned side slide
    # so jaws straddle the handle without crushing the tines.
    "matcha_whisk": {
        "approach_height_offset": 0.15,   # extra clearance to avoid the tall tine bundle
        "grasp_height_offset":    0.04,   # grip 4 cm above centroid (mid-handle region)
        "wrist_angle":           15.0,    # tilt into the handle's natural lean angle
        "grip_width":             0.03,   # narrow handle (~2.5 cm); close firmly
        "object_height":          0.17,   # tines add significant vertical extent
        "approach_axis":          "yaw_aligned",
        "approach_standoff":      0.06,   # slide in from 6 cm off the handle axis
    },

    # ---------------------------------------------------------------- matcha_scoop
    # Small bamboo scoop, ~14 cm handle, narrow.
    # Lies flat; grip near the base of the handle so the scoop head stays level
    # and powder doesn't spill during the lift.
    "matcha_scoop": {
        "approach_height_offset": 0.10,
        "grasp_height_offset":    0.01,   # grasp near table level — handle lies flat
        "wrist_angle":           10.0,    # slight tilt to match the scoop's resting angle
        "grip_width":             0.025,  # very narrow handle
        "object_height":          0.02,
        "approach_axis":          "yaw_aligned",
        "approach_standoff":      0.05,
    },

    # ------------------------------------------------------------------ cup_of_ice
    # A cup filled with ice — heavier than it looks.
    # Grip lower on the body (not near the rim) for a stable centre of mass.
    "cup_of_ice": {
        "approach_height_offset": 0.12,
        "grasp_height_offset":    0.03,   # grip lower on body for stability under load
        "wrist_angle":            0.0,    # top-down: symmetric grip on cylinder
        "grip_width":             0.07,
        "object_height":          0.12,
        "approach_axis":          "y",
        "approach_standoff":      0.0,
    },

    # ----------------------------------------------------------------- cup_of_water
    # Same geometry as the ice cup.  Approach slowly (executor-level speed
    # control) to avoid sloshing; grip geometry is identical.
    "cup_of_water": {
        "approach_height_offset": 0.12,
        "grasp_height_offset":    0.03,   # same as ice cup — matching vessel
        "wrist_angle":            0.0,
        "grip_width":             0.07,
        "object_height":          0.12,
        "approach_axis":          "y",
        "approach_standoff":      0.0,
    },
}

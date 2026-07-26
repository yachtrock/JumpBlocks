"""Author the JumpBlocks player character: a chibi plumber with rig + anims.

Run headless:
    blender --background --python tools/build_player_model.py

Outputs:
    assets/models/player.blend  — editable source (mesh, armature, actions)
    assets/models/player.glb    — game asset (skinned, 5 named animations)

Design notes for tuning in Blender:
- One mesh ("PlayerMesh"), skinned with RIGID weights: every vertex belongs
  100% to a single bone's vertex group (named after the bone). Repaint or
  soften weights freely — the exporter just reads the groups.
- Armature "PlayerRig", 14 deform bones:
      root
        hips
          chest
            head
            upper_arm.L/R -> forearm.L/R
          thigh.L/R -> shin.L/R -> foot.L/R
- Each animation is an Action with a fake user ("Idle", "Walk", "Run",
  "Jump", "Fall"). The glTF exporter emits every action as a named
  animation. Frame rate 30fps; all cycles loop except "Jump".
- Character faces -Y in Blender == -Z in the game (glTF Y-up export).
"""

import math
import os

import bpy
from mathutils import Vector

FPS = 30
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "models")

# ---------------------------------------------------------------------------
# Scene reset
# ---------------------------------------------------------------------------

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.fps = FPS

# ---------------------------------------------------------------------------
# Materials (simple flat Principled colors)
# ---------------------------------------------------------------------------

def make_mat(name, rgb):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.85
    return mat

MAT_RED = make_mat("Red", (0.78, 0.09, 0.06))        # cap + shirt
MAT_BLUE = make_mat("Blue", (0.10, 0.22, 0.70))      # overalls
MAT_SKIN = make_mat("Skin", (0.92, 0.66, 0.45))
MAT_BROWN = make_mat("Brown", (0.28, 0.14, 0.06))    # shoes, hair, mustache
MAT_WHITE = make_mat("White", (0.92, 0.92, 0.92))    # gloves
MAT_DARK = make_mat("Dark", (0.05, 0.04, 0.04))      # eyes
MAT_YELLOW = make_mat("Yellow", (0.95, 0.78, 0.10))  # buttons

# ---------------------------------------------------------------------------
# Mesh parts — each part is created, assigned a material and a target bone.
# `parts` collects (object, bone_name); vertex groups are added before join.
# ---------------------------------------------------------------------------

parts = []

def add_part(obj, mat, bone):
    obj.data.materials.append(mat)
    parts.append((obj, bone))
    return obj

def sphere(name, r, loc, scale=(1, 1, 1), mat=MAT_SKIN, bone="chest", segments=24, rings=16):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=loc, segments=segments, ring_count=rings)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.object.shade_smooth()
    return add_part(obj, mat, bone)

def cylinder(name, r, z0, z1, x, mat, bone, y=0.0):
    h = z1 - z0
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=h, location=(x, y, (z0 + z1) / 2), vertices=16)
    obj = bpy.context.active_object
    obj.name = name
    bpy.ops.object.shade_smooth()
    return add_part(obj, mat, bone)

def box(name, size, loc, mat, bone):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (size[0] / 2, size[1] / 2, size[2] / 2)
    bpy.ops.object.transform_apply(scale=True)
    return add_part(obj, mat, bone)

# Torso
sphere("Hips", 0.23, (0, 0, 0.50), scale=(1.0, 0.92, 0.80), mat=MAT_BLUE, bone="hips")
sphere("Chest", 0.215, (0, 0, 0.68), scale=(1.0, 0.88, 0.82), mat=MAT_RED, bone="chest")
# Overall bib + straps (front plate)
box("Bib", (0.20, 0.045, 0.14), (0, -0.185, 0.70), MAT_BLUE, "chest")
box("Strap.L", (0.05, 0.045, 0.16), (0.075, -0.16, 0.80), MAT_BLUE, "chest")
box("Strap.R", (0.05, 0.045, 0.16), (-0.075, -0.16, 0.80), MAT_BLUE, "chest")
sphere("Button.L", 0.028, (0.075, -0.20, 0.755), mat=MAT_YELLOW, bone="chest", segments=12, rings=8)
sphere("Button.R", 0.028, (-0.075, -0.20, 0.755), mat=MAT_YELLOW, bone="chest", segments=12, rings=8)

# Head + face + cap
sphere("Head", 0.30, (0, 0, 1.08), scale=(1.0, 0.95, 0.92), mat=MAT_SKIN, bone="head")
sphere("Nose", 0.062, (0, -0.285, 1.05), mat=MAT_SKIN, bone="head", segments=16, rings=12)
box("Mustache", (0.19, 0.05, 0.05), (0, -0.262, 0.985), MAT_BROWN, "head")
sphere("Eye.L", 0.034, (0.085, -0.255, 1.13), mat=MAT_DARK, bone="head", segments=12, rings=8)
sphere("Eye.R", 0.034, (-0.085, -0.255, 1.13), mat=MAT_DARK, bone="head", segments=12, rings=8)
sphere("Ear.L", 0.05, (0.285, 0.02, 1.06), mat=MAT_SKIN, bone="head", segments=12, rings=8)
sphere("Ear.R", 0.05, (-0.285, 0.02, 1.06), mat=MAT_SKIN, bone="head", segments=12, rings=8)
# Hair back
sphere("Hair", 0.285, (0, 0.06, 1.04), scale=(0.98, 0.95, 0.85), mat=MAT_BROWN, bone="head")
# Cap dome + brim
sphere("Cap", 0.315, (0, 0.01, 1.185), scale=(1.0, 0.98, 0.62), mat=MAT_RED, bone="head")
box("Brim", (0.34, 0.18, 0.035), (0, -0.36, 1.20), MAT_RED, "head")

# Arms (hang straight down at the sides)
cylinder("UpperArm.L", 0.055, 0.585, 0.755, 0.265, MAT_RED, "upper_arm.L")
cylinder("UpperArm.R", 0.055, 0.585, 0.755, -0.265, MAT_RED, "upper_arm.R")
sphere("Shoulder.L", 0.07, (0.265, 0, 0.74), mat=MAT_RED, bone="upper_arm.L", segments=12, rings=8)
sphere("Shoulder.R", 0.07, (-0.265, 0, 0.74), mat=MAT_RED, bone="upper_arm.R", segments=12, rings=8)
cylinder("Forearm.L", 0.05, 0.46, 0.60, 0.265, MAT_RED, "forearm.L")
cylinder("Forearm.R", 0.05, 0.46, 0.60, -0.265, MAT_RED, "forearm.R")
sphere("Glove.L", 0.082, (0.265, 0, 0.415), mat=MAT_WHITE, bone="forearm.L", segments=16, rings=12)
sphere("Glove.R", 0.082, (-0.265, 0, 0.415), mat=MAT_WHITE, bone="forearm.R", segments=16, rings=12)

# Legs
cylinder("Thigh.L", 0.068, 0.27, 0.44, 0.105, MAT_BLUE, "thigh.L")
cylinder("Thigh.R", 0.068, 0.27, 0.44, -0.105, MAT_BLUE, "thigh.R")
cylinder("Shin.L", 0.058, 0.115, 0.28, 0.105, MAT_BLUE, "shin.L")
cylinder("Shin.R", 0.058, 0.115, 0.28, -0.105, MAT_BLUE, "shin.R")
box("Shoe.L", (0.15, 0.27, 0.115), (0.105, -0.045, 0.058), MAT_BROWN, "foot.L")
box("Shoe.R", (0.15, 0.27, 0.115), (-0.105, -0.045, 0.058), MAT_BROWN, "foot.R")

# --- Assign one full-weight vertex group per part, then join -----------------

for obj, bone in parts:
    vg = obj.vertex_groups.new(name=bone)
    vg.add(range(len(obj.data.vertices)), 1.0, "REPLACE")

bpy.ops.object.select_all(action="DESELECT")
for obj, _ in parts:
    obj.select_set(True)
body = parts[0][0]
bpy.context.view_layer.objects.active = body
bpy.ops.object.join()
body.name = "PlayerMesh"
body.data.name = "PlayerMesh"

# ---------------------------------------------------------------------------
# Armature
# ---------------------------------------------------------------------------

bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
rig = bpy.context.active_object
rig.name = "PlayerRig"
arm = rig.data
arm.name = "PlayerRig"

eb = arm.edit_bones
eb.remove(eb[0])

def bone(name, head, tail, parent=None):
    b = eb.new(name)
    b.head = Vector(head)
    b.tail = Vector(tail)
    b.roll = 0.0
    if parent:
        b.parent = eb[parent]
    return b

bone("root", (0, 0, 0), (0, 0.18, 0))
bone("hips", (0, 0, 0.44), (0, 0, 0.58), "root")
bone("chest", (0, 0, 0.58), (0, 0, 0.82), "hips")
bone("head", (0, 0, 0.84), (0, 0, 1.30), "chest")
for side, sx in (("L", 1.0), ("R", -1.0)):
    bone(f"upper_arm.{side}", (sx * 0.265, 0, 0.755), (sx * 0.265, 0, 0.60), "chest")
    bone(f"forearm.{side}", (sx * 0.265, 0, 0.60), (sx * 0.265, 0, 0.42), f"upper_arm.{side}")
    bone(f"thigh.{side}", (sx * 0.105, 0, 0.45), (sx * 0.105, 0, 0.28), "hips")
    bone(f"shin.{side}", (sx * 0.105, 0, 0.28), (sx * 0.105, 0, 0.115), f"thigh.{side}")
    bone(f"foot.{side}", (sx * 0.105, 0, 0.115), (sx * 0.105, -0.19, 0.04), f"shin.{side}")

bpy.ops.object.mode_set(mode="OBJECT")

# Parent mesh to armature, keeping our rigid vertex groups.
bpy.ops.object.select_all(action="DESELECT")
body.select_set(True)
rig.select_set(True)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.parent_set(type="ARMATURE")

# ---------------------------------------------------------------------------
# Animation authoring
# ---------------------------------------------------------------------------
#
# Pose-bone local axes with roll 0:
#  - vertical bones (hips/chest/head): +X rotation = lean forward
#  - hanging bones (arms/legs):        -X rotation = swing forward (-Y world)
#  - shins: +X = knee bend (heel back);  forearms: -X = elbow curl

D = math.radians

def set_key(action_holder, bone_name, frame, rot=None, loc=None):
    pb = rig.pose.bones[bone_name]
    pb.rotation_mode = "XYZ"
    if rot is not None:
        pb.rotation_euler = rot
        pb.keyframe_insert("rotation_euler", frame=frame)
    if loc is not None:
        pb.location = loc
        pb.keyframe_insert("location", frame=frame)

def reset_pose():
    for pb in rig.pose.bones:
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = (0, 0, 0)
        pb.location = (0, 0, 0)

def new_action(name, start, end):
    action = bpy.data.actions.new(name)
    action.use_fake_user = True
    if rig.animation_data is None:
        rig.animation_data_create()
    rig.animation_data.action = action
    scene.frame_start = start
    scene.frame_end = end
    reset_pose()
    return action

def zero_all(frame):
    """Key every bone at rest so each action fully overrides the previous."""
    for pb in rig.pose.bones:
        set_key(None, pb.name, frame, rot=(0, 0, 0), loc=(0, 0, 0))

# --- Idle: gentle breathing sway (2s loop) ---------------------------------

new_action("Idle", 1, 61)
zero_all(1)
for f, t in ((1, 0.0), (31, 1.0), (61, 0.0)):
    s = math.sin(t * math.pi)  # 0 → 1 → 0
    set_key(None, "hips", f, loc=(0, -0.012 * s, 0))
    set_key(None, "chest", f, rot=(D(2.5) * s, 0, 0))
    set_key(None, "head", f, rot=(D(-2.0) * s, 0, 0))
    set_key(None, "upper_arm.L", f, rot=(D(3.0) * s, 0, D(-2.0) * s))
    set_key(None, "upper_arm.R", f, rot=(D(3.0) * s, 0, D(2.0) * s))

# --- Walk: relaxed stride (~1.07s loop) ------------------------------------

def stride(name, frames, thigh, knee, arm, elbow, torso, bob, head_pitch):
    """Symmetric two-beat locomotion cycle over `frames` (loop frame = end)."""
    new_action(name, 1, frames + 1)
    zero_all(1)
    half = frames // 2
    for phase_start, sign in ((1, 1.0), (half + 1, -1.0)):
        # Contact pose at phase start: L forward when sign=+1
        set_key(None, "thigh.L", phase_start, rot=(-D(thigh) * sign, 0, 0))
        set_key(None, "thigh.R", phase_start, rot=(D(thigh) * sign, 0, 0))
        # Knee: the swinging (back) leg folds mid-phase
        mid = phase_start + half // 2
        back = "shin.L" if sign < 0 else "shin.R"
        fwd = "shin.L" if sign > 0 else "shin.R"
        set_key(None, back, phase_start, rot=(D(knee) * 0.35, 0, 0))
        set_key(None, back, mid, rot=(D(knee), 0, 0))
        set_key(None, fwd, phase_start, rot=(D(knee) * 0.1, 0, 0))
        set_key(None, fwd, mid, rot=(D(knee) * 0.1, 0, 0))
        # Arms counter-swing, slight constant elbow curl
        set_key(None, "upper_arm.L", phase_start, rot=(D(arm) * sign, 0, D(-3)))
        set_key(None, "upper_arm.R", phase_start, rot=(-D(arm) * sign, 0, D(3)))
        set_key(None, "forearm.L", phase_start, rot=(-D(elbow), 0, 0))
        set_key(None, "forearm.R", phase_start, rot=(-D(elbow), 0, 0))
        # Torso lean + double bob (down at contact, up mid-stride)
        set_key(None, "chest", phase_start, rot=(D(torso), 0, 0))
        set_key(None, "head", phase_start, rot=(D(-torso * 0.6) + D(head_pitch), 0, 0))
        set_key(None, "hips", phase_start, loc=(0, -bob, 0))
        set_key(None, "hips", mid, loc=(0, bob * 0.4, 0))
        # Feet: toe-off flex on the back foot
        set_key(None, f"foot.L", phase_start, rot=(D(10) * max(sign, 0), 0, 0))
        set_key(None, f"foot.R", phase_start, rot=(D(10) * max(-sign, 0), 0, 0))
    # Loop closure = copy of frame 1 pose
    set_key(None, "thigh.L", frames + 1, rot=(-D(thigh), 0, 0))
    set_key(None, "thigh.R", frames + 1, rot=(D(thigh), 0, 0))
    set_key(None, "shin.R", frames + 1, rot=(D(knee) * 0.35, 0, 0))
    set_key(None, "shin.L", frames + 1, rot=(D(knee) * 0.1, 0, 0))
    set_key(None, "upper_arm.L", frames + 1, rot=(D(arm), 0, D(-3)))
    set_key(None, "upper_arm.R", frames + 1, rot=(-D(arm), 0, D(3)))
    set_key(None, "forearm.L", frames + 1, rot=(-D(elbow), 0, 0))
    set_key(None, "forearm.R", frames + 1, rot=(-D(elbow), 0, 0))
    set_key(None, "chest", frames + 1, rot=(D(torso), 0, 0))
    set_key(None, "head", frames + 1, rot=(D(-torso * 0.6) + D(0), 0, 0))
    set_key(None, "hips", frames + 1, loc=(0, -bob, 0))
    set_key(None, "foot.L", frames + 1, rot=(D(10), 0, 0))
    set_key(None, "foot.R", frames + 1, rot=(0, 0, 0))

stride("Walk", 32, thigh=27, knee=40, arm=18, elbow=15, torso=4, bob=0.015, head_pitch=0)
stride("Run", 20, thigh=52, knee=70, arm=42, elbow=55, torso=13, bob=0.03, head_pitch=4)

# --- Jump: the classic one-fist-up leap (one shot) -------------------------

new_action("Jump", 1, 25)
zero_all(1)
# Anticipation crouch is handled by gameplay squash; go straight to the pose.
set_key(None, "chest", 6, rot=(D(-6), 0, 0))
set_key(None, "head", 6, rot=(D(4), 0, 0))
set_key(None, "upper_arm.R", 6, rot=(D(-165), 0, D(8)))     # fist to the sky
set_key(None, "forearm.R", 6, rot=(D(-10), 0, 0))
set_key(None, "upper_arm.L", 6, rot=(D(35), 0, D(-12)))     # trailing arm
set_key(None, "forearm.L", 6, rot=(D(-25), 0, 0))
set_key(None, "thigh.L", 6, rot=(D(-55), 0, 0))             # lead knee tucked
set_key(None, "shin.L", 6, rot=(D(75), 0, 0))
set_key(None, "thigh.R", 6, rot=(D(25), 0, 0))              # push leg back
set_key(None, "shin.R", 6, rot=(D(20), 0, 0))
set_key(None, "foot.L", 6, rot=(D(15), 0, 0))
set_key(None, "foot.R", 6, rot=(D(25), 0, 0))
# Hold the pose
for b in ("chest", "head", "upper_arm.R", "forearm.R", "upper_arm.L",
          "forearm.L", "thigh.L", "shin.L", "thigh.R", "shin.R",
          "foot.L", "foot.R"):
    pb = rig.pose.bones[b]
    pb.keyframe_insert("rotation_euler", frame=25)

# --- Fall: arms up, legs pedaling (loop) -----------------------------------

new_action("Fall", 1, 41)
zero_all(1)
for f, t in ((1, 0.0), (11, 0.5), (21, 1.0), (31, 0.5), (41, 0.0)):
    w = math.sin(t * math.pi)
    set_key(None, "chest", f, rot=(D(8), 0, 0))
    set_key(None, "head", f, rot=(D(-10), 0, 0))
    set_key(None, "upper_arm.L", f, rot=(D(-150) + D(12) * w, 0, D(-18)))
    set_key(None, "upper_arm.R", f, rot=(D(-150) - D(12) * w, 0, D(18)))
    set_key(None, "forearm.L", f, rot=(D(-15), 0, 0))
    set_key(None, "forearm.R", f, rot=(D(-15), 0, 0))
    set_key(None, "thigh.L", f, rot=(D(-25) + D(22) * w, 0, 0))
    set_key(None, "thigh.R", f, rot=(D(-3) - D(22) * w, 0, 0))
    set_key(None, "shin.L", f, rot=(D(35), 0, 0))
    set_key(None, "shin.R", f, rot=(D(45), 0, 0))

# Leave Idle as the active action.
rig.animation_data.action = bpy.data.actions["Idle"]
reset_pose()

# ---------------------------------------------------------------------------
# Save + export
# ---------------------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
blend_path = os.path.abspath(os.path.join(OUT_DIR, "player.blend"))
glb_path = os.path.abspath(os.path.join(OUT_DIR, "player.glb"))

bpy.ops.wm.save_as_mainfile(filepath=blend_path)
bpy.ops.export_scene.gltf(
    filepath=glb_path,
    export_format="GLB",
    export_animations=True,
    export_skins=True,
    export_yup=True,
    export_apply=False,
)

print(f"Wrote {blend_path}")
print(f"Wrote {glb_path}")

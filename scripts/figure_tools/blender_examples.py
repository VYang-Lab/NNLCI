"""Small CPU-only render smoke tests, reserved for presentation rendering."""
from pathlib import Path
import os
import math
import bpy
from mathutils import Vector

out = Path(os.environ['NNLCI_FIGURE_OUTPUT'])
for name in ('layers', 'samples'):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 8
    scene.cycles.use_denoising = False  # Debian build has no OpenImageDenoise.
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 2
    scene.render.resolution_x, scene.render.resolution_y = 640, 400
    scene.render.resolution_percentage = 100
    scene.world = bpy.data.worlds.new('WhiteWorld')
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs[0].default_value = (.8, .85, .9, 1)
    material = bpy.data.materials.new('Teal')
    material.diffuse_color = (.025, .4, .46, 1)
    for i in range(3):
        if name == 'layers':
            bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, i*.6))
            bpy.context.object.scale = (2.2, 1.6, .12)
        else:
            bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=.28, location=(i*.8-.8, 0, .5))
        bpy.context.object.data.materials.append(material)
    bpy.ops.object.light_add(type='AREA', location=(0, -3, 5))
    bpy.context.object.data.energy, bpy.context.object.data.size = 400, 4
    bpy.ops.object.camera_add(location=(4, -6, 4))
    camera = bpy.context.object
    camera.rotation_euler = (Vector((0, 0, .7))-camera.location).to_track_quat('-Z', 'Y').to_euler()
    camera.data.type, camera.data.ortho_scale = 'ORTHO', 4
    scene.camera = camera
    scene.render.filepath = str(out / f'blender_{name}.png')
    bpy.ops.wm.save_as_mainfile(filepath=str(out / f'blender_{name}.blend'))
    bpy.ops.render.render(write_still=True)

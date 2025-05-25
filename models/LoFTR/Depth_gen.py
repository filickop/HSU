import bpy
import os
import numpy as np

# Nastavenie cesty
output_dir = "D:/SKOLA/HSU/SEM/dataset"
os.makedirs(f"{output_dir}/images", exist_ok=True)
os.makedirs(f"{output_dir}/depth", exist_ok=True)
os.makedirs(f"{output_dir}/intrinsics", exist_ok=True)
os.makedirs(f"{output_dir}/poses", exist_ok=True)

# Nastavenie scény
scene = bpy.context.scene
scene.render.image_settings.file_format = 'PNG'

# Kamera
camera = bpy.data.objects['Camera']
camera.location = (0, -3, 1)
camera.rotation_euler = (1.1, 0, 0)

# Pridáme kocku (objekt)
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))

# Vypneme antialiasing pre depth
scene.render.use_multiview = False
scene.render.image_settings.color_depth = '8'

# Aktivuj depth výstup
scene.use_nodes = True
tree = scene.node_tree
links = tree.links

for node in tree.nodes:
    tree.nodes.remove(node)

rlayers = tree.nodes.new('CompositorNodeRLayers')
depth_out = tree.nodes.new(type="CompositorNodeOutputFile")
depth_out.base_path = f"{output_dir}/depth"
depth_out.file_slots[0].path = "depth_####"

links.new(rlayers.outputs['Depth'], depth_out.inputs[0])

# Renderuj niekoľko snímok z rôznych pohľadov
for frame in range(5):
    camera.location.x += 0.1  # Mierne posúvame kameru
    scene.render.filepath = f"{output_dir}/images/image_{frame:03d}.png"
    bpy.ops.render.render(write_still=True)

    # Ulož intrinzické parametre a pózu
    K = camera.calc_matrix_camera(
        bpy.context.evaluated_depsgraph_get()).to_3x3().transposed()
    np.savetxt(f"{output_dir}/intrinsics/image_{frame:03d}.txt", K)

    pose = camera.matrix_world.inverted()
    np.savetxt(f"{output_dir}/poses/image_{frame:03d}.txt", pose)

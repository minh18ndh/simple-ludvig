import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud(
    "./outputs/semantic_bonsai.ply"
)

points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

print("Points:", points.shape)
print("Colors:", colors.shape)

print("Color min:", colors.min())
print("Color max:", colors.max())
print("Color mean:", colors.mean())

# Force brighter colors
colors = np.clip(colors * 2.0, 0, 1)

pcd.colors = o3d.utility.Vector3dVector(colors)

vis = o3d.visualization.Visualizer()

vis.create_window(
    width=1600,
    height=900
)

vis.add_geometry(pcd)

opt = vis.get_render_option()

opt.point_size = 1.0

opt.light_on = False

opt.background_color = np.array([1, 1, 1])

ctr = vis.get_view_control()

ctr.set_zoom(0.7)

vis.run()

vis.destroy_window()
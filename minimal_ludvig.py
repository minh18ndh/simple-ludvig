import os
import json
import numpy as np
import open3d as o3d
import torch

from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor, AutoModel

# ============================================================
# CONFIG
# ============================================================

IMAGE_DIR = "./360_v2/bonsai/images"

PLY_PATH = "./models/bonsai/point_cloud/iteration_30000/point_cloud.ply"

CAMERA_JSON = "./models/bonsai/cameras.json"

OUTPUT_PLY = "./outputs/semantic_bonsai.ply"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("./outputs", exist_ok=True)

# ============================================================
# LOAD DINOv2
# ============================================================

print("Loading DINOv2...")

processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov2-base"
)

model = AutoModel.from_pretrained(
    "facebook/dinov2-base"
).to(DEVICE)

model.eval()

# ============================================================
# LOAD POINT CLOUD
# ============================================================

print("Loading point cloud...")

pcd = o3d.io.read_point_cloud(PLY_PATH)

points = np.asarray(pcd.points)

N = len(points)

print(f"Loaded {N} points")

# ============================================================
# LOAD CAMERA DATA
# ============================================================

print("Loading cameras...")

with open(CAMERA_JSON, "r") as f:
    cameras = json.load(f)

# ============================================================
# FEATURE STORAGE
# ============================================================

feature_dim = 768

point_features = np.zeros((N, feature_dim), dtype=np.float32)

point_counts = np.zeros(N, dtype=np.float32)

# ============================================================
# PROJECTION
# ============================================================

def project_points(points, K, R, t):

    # World -> Camera

    pts_cam = (R @ (points - t).T).T

    z = pts_cam[:, 2]

    valid = z > 0

    pts_cam = pts_cam[valid]

    indices = np.where(valid)[0]

    pts_2d = (K @ pts_cam.T).T

    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]

    return pts_2d, indices

# ============================================================
# MAIN LOOP
# ============================================================

print("Uplifting semantic features...")

for cam_id, cam in enumerate(tqdm(cameras)):

    image_name = cam["img_name"] + ".JPG"

    image_path = os.path.join(IMAGE_DIR, image_name)

    if not os.path.exists(image_path):

        print("Missing:", image_path)

        continue

    print(f"\nProcessing {image_name}")

    image = Image.open(image_path).convert("RGB")

    width, height = image.size

    # --------------------------------------------------------
    # DINO FEATURES
    # --------------------------------------------------------

    inputs = processor(images=image, return_tensors="pt")

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():

        outputs = model(**inputs)

        tokens = outputs.last_hidden_state[0]

    # Remove CLS token

    tokens = tokens[1:]

    patch_size = 14

    num_tokens = tokens.shape[0]

    grid_size = int(np.sqrt(num_tokens))

    feat_h = grid_size
    feat_w = grid_size

    feat_map = tokens.reshape(
        feat_h,
        feat_w,
        feature_dim
    )

    feat_map = feat_map.cpu().numpy()

    print("Feature map:", feat_map.shape)

    # --------------------------------------------------------
    # CAMERA MATRICES
    # --------------------------------------------------------

    fx = cam["fx"]
    fy = cam["fy"]

    cx = width / 2
    cy = height / 2

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    R = np.array(cam["rotation"])

    t = np.array(cam["position"])

    # --------------------------------------------------------
    # PROJECT POINTS
    # --------------------------------------------------------

    pts_2d, valid_idx = project_points(
        points,
        K,
        R,
        t
    )

    print("Projected points:", len(valid_idx))

    # --------------------------------------------------------
    # SAMPLE FEATURES
    # --------------------------------------------------------

    assigned = 0

    for pixel, point_idx in zip(pts_2d, valid_idx):

        x, y = pixel

        px = int((x / width) * feat_w)
        py = int((y / height) * feat_h)

        if (
            px < 0
            or px >= feat_w
            or py < 0
            or py >= feat_h
        ):
            continue

        feat = feat_map[py, px]

        point_features[point_idx] += feat

        point_counts[point_idx] += 1

        assigned += 1

    print("Assigned features:", assigned)

# ============================================================
# NORMALIZE
# ============================================================

print("\nNormalizing features...")

valid = point_counts > 10

print("Valid points:", valid.sum())
print("Counts stats:")
print("Min:", point_counts[valid].min())
print("Max:", point_counts[valid].max())
print("Mean:", point_counts[valid].mean())

if valid.sum() == 0:

    raise RuntimeError(
        "No points received features."
    )

point_features[valid] /= point_counts[valid][:, None]

# ============================================================
# PCA VISUALIZATION
# ============================================================

print("Computing PCA colors...")

pca = PCA(n_components=3)

reduced = pca.fit_transform(point_features[valid])

reduced -= reduced.min(0)

reduced /= reduced.max(0) + 1e-8

colors = np.zeros((N, 3))

colors[valid] = reduced
print("Color stats:")
print(colors.min())
print(colors.max())
print(colors.mean())

pcd.colors = o3d.utility.Vector3dVector(colors)

# ============================================================
# SAVE RESULT
# ============================================================

print("Saving semantic point cloud...")

o3d.io.write_point_cloud(
    OUTPUT_PLY,
    pcd
)

print("\nDONE")
print("Saved to:", OUTPUT_PLY)
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

left_iris_index = {474, 475, 476, 477}
right_iris_index = {469, 470, 471, 472}
left_eye_index = {
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
}
right_eye_index = {
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,
}
facemesh_index = (
    {_ for _ in range(478)}
    - left_iris_index
    - right_iris_index
    - left_eye_index
    - right_eye_index
)

iris_idx = sorted(list(left_iris_index | right_iris_index))
eye_idx = sorted(list(left_eye_index | right_eye_index))
face_idx = sorted(list(facemesh_index))

iris_columns = [f"iris{e}_{c}" for e in iris_idx for c in ["x", "y", "z"]]
eye_columns = [f"eye{e}_{c}" for e in iris_idx for c in ["x", "y", "z"]]
face_columns = [f"face{e}_{c}" for e in iris_idx for c in ["x", "y", "z"]]


def get_facemesh_coords(
    landmark_list: NormalizedLandmarkList, img: np.ndarray, normalize: bool = True
) -> np.ndarray:
    """Extract FaceMesh landmark coordinates into 468x3 NumPy array."""
    h, w = img.shape[:2]  # grab width and height from image
    xyz = [(lm.x, lm.y, lm.z) for lm in landmark_list.landmark]
    if normalize:
        return np.array(xyz).astype(float)
    return np.multiply(xyz, [w, h, w]).astype(int)

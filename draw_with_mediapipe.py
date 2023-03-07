import csv
import os

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm


def get_facemesh_coords(
    landmark_list, img: np.ndarray, normalize: bool = True
) -> np.ndarray:
    """Extract FaceMesh landmark coordinates into 468x3 NumPy array."""
    h, w = img.shape[:2]  # grab width and height from image
    xyz = [(lm.x, lm.y, lm.z) for lm in landmark_list.landmark]
    if normalize:
        return np.array(xyz).astype(float)
    return np.multiply(xyz, [w, h, w]).astype(int)


def search_set(landmark, element) -> None:
    for t in landmark:
        if element in t:
            print(t)


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

write_results = []
write_results.append(["PATH", "IRIS", "CONTOURS", "FACE_BB", "EYE_BB", "IRIS_BB"])
dirs = os.listdir("./photographed_data")
for dir in tqdm(dirs):
    os.makedirs(f"./drawing_data/{dir}", exist_ok=True)
    photos = os.listdir(f"./photographed_data/{dir}")
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        for photo in tqdm(photos, leave=False):
            path = f"./photographed_data/{dir}/{photo}"
            image = cv2.imread(f"./photographed_data/{dir}/{photo}")
            image.flags.writeable = False  # こうするとはやいらしい
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # results.multi_face_landmarksは多分検出人数分のリスト
            if results.multi_face_landmarks is not None:
                face_landmarks = results.multi_face_landmarks[0]
                # print(mp_face_mesh.FACEMESH_IRISES)
                # print(mp_face_mesh.FACEMESH_CONTOURS)
                """
                contours = set()
                for tmp in mp_face_mesh.FACEMESH_CONTOURS:
                    for t in tmp:
                        contours.add(t)
                """
                # 目の虹彩のキーポイント
                iris = {469, 470, 471, 472, 474, 475, 476, 477}
                # 目の輪郭のキーポイント
                # 163から7まで左目 以降右目
                contours_eye = {
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
                    33,
                    7,
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
                    362,
                    382,
                    381,
                }
                # 生の座標を取得する
                coords = get_facemesh_coords(face_landmarks, image, normalize=False)
                iris_coords = []
                contours_eye_coords = []
                for e in iris:
                    image = cv2.circle(
                        image, (coords[e, 0], coords[e, 1]), 2, (255, 0, 0)
                    )
                    # image = cv2.putText(image, str(e),(coords[e,0], coords[e,1]),cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 0, 0))
                    iris_coords.append(f"{coords[e,0]} {coords[e,1]} {coords[e,2]}")
                for e in contours_eye:
                    image = cv2.circle(
                        image, (coords[e, 0], coords[e, 1]), 2, (255, 0, 0)
                    )
                    # image = cv2.putText(image, str(e),(coords[e,0], coords[e,1]),cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 0, 0))
                    contours_eye_coords.append(
                        f"{coords[e,0]} {coords[e,1]} {coords[e,2]}"
                    )
                # 顔のバウンディングボックスの位置
                left = coords[123, 0]
                top = coords[10, 1]
                right = coords[352, 0]
                bottom = coords[152, 1]
                bb_coords = f"{left} {top} {right} {bottom}"
                cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0))
                # 目の輪郭のバウンディングボックスの位置
                left = coords[33, 0]
                top = min(coords[470, 1], coords[475, 1])
                right = coords[263, 0]
                bottom = max(coords[477, 1], coords[472, 1])
                eye_bb_coords = f"{left} {top} {right} {bottom}"
                cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0))
                # 目の虹彩のバウンディングボックスの位置
                left = coords[471, 0]
                top = min(coords[470, 1], coords[475, 1])
                right = coords[474, 0]
                bottom = max(coords[477, 1], coords[472, 1])
                iris_bb_coords = f"{left} {top} {right} {bottom}"

            else:  # 検出が得られない場合
                iris_coords = "000"
                contours_eye = "000"
                contours_eye_coords = "000"
                bb_coords = "000"
                eye_bb_coords = "000"
                iris_bb_coords = "000"
            write_result = [
                path,
                iris_coords,
                contours_eye_coords,
                bb_coords,
                eye_bb_coords,
                iris_bb_coords,
            ]
            write_results.append(write_result)
            cv2.imwrite(f"./drawing_data/{dir}/{photo}", image)

with open("./keypoint.csv", "w") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerows(write_results)

from __future__ import annotations
import cv2
import torch
import numpy as np
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import ObservationTermCfg
from omni.isaac.lab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv

import torch
import cv2
import numpy as np
import os
from collections import deque

# # 🔥 64개 배치 각각의 프레임을 저장하는 큐 (각 배치에 대해 5개 저장)
# frame_queues = [deque(maxlen=5) for _ in range(64)]

# def image_contour_debug(
#     env: ManagerBasedEnv,
#     sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
#     data_type: str = "rgb",
#     convert_perspective_to_orthogonal: bool = False,
#     normalize: bool = True
# ) -> torch.Tensor:
#     sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]
#     images = sensor.data.output[data_type]  

#     if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
#         images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)

#     if normalize and data_type == "rgb":
#         images = images.float() / 255.0
#         mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
#         images -= mean_tensor
#     elif "distance_to" in data_type or "depth" in data_type:
#         images[images == float("inf")] = 0

#     if data_type == "rgb":
#         img_np = images.cpu().numpy()

#         img_np = ((img_np + 1) * 127.5).astype(np.uint8)

#         img_gray = np.zeros((img_np.shape[0], img_np.shape[1], img_np.shape[2]), dtype=np.uint8)
#         for i in range(img_np.shape[0]):
#             img_gray[i] = cv2.cvtColor(img_np[i], cv2.COLOR_RGB2GRAY)

#         enhanced_images = np.zeros_like(img_gray)
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#         for i in range(img_gray.shape[0]):
#             enhanced_images[i] = clahe.apply(img_gray[i])

#         edge_images = np.zeros_like(enhanced_images)
#         for i in range(enhanced_images.shape[0]):
#             edge_images[i] = cv2.Canny(enhanced_images[i], 20, 40)

#         # 🔹 큐에 프레임 추가
#         for i in range(64):
#             frame_queues[i].append(edge_images[i])

#         # 🔹 64개의 배치에 대해 5개 프레임을 스택 형태로 변환 (64, 5, height, width)
#         stacked_frames = np.zeros((64, 5, edge_images.shape[1], edge_images.shape[2]), dtype=np.uint8)
#         for i in range(64):
#             frames_list = list(frame_queues[i])
#             for j, frame in enumerate(frames_list):
#                 stacked_frames[i, j] = frame

#         # 🔹 터미널에 ASCII 출력 (최신 프레임의 Batch 0만 표시)
#         os.system('clear' if os.name == 'posix' else 'cls') 
#         print("\n🖥️ Contour Detection (ASCII View)\n" + "="*40)

#         ascii_chars = ['.', '#']  
#         resized = cv2.resize(stacked_frames[0, -1], (80, 40))  
#         ascii_img = '\n'.join(
#             ''.join(ascii_chars[1] if pixel > 0 else ascii_chars[0] for pixel in row)
#             for row in resized
#         )
#         print(ascii_img)  

#         # 🔹 PyTorch 텐서로 변환 (64, 5, height * width)
#         contour_tensor = torch.from_numpy(stacked_frames).float() / 255.0
#         contour_tensor = contour_tensor.to(images.device)
#         reshaped_tensor = contour_tensor.view(64,-1)  

#         return reshaped_tensor

#     return images
frame_queues = [deque(maxlen=28) for _ in range(64)]

def image_contour_debug(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True
) -> torch.Tensor:
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]
    images = sensor.data.output[data_type]  

    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
        images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)

    if normalize and data_type == "rgb":
        images = images.float() / 255.0
        mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
        images -= mean_tensor
    elif "distance_to" in data_type or "depth" in data_type:
        images[images == float("inf")] = 0

    if data_type == "rgb":
        img_np = images.cpu().numpy()
        img_np = ((img_np + 1) * 127.5).astype(np.uint8)

        img_gray = np.zeros((img_np.shape[0], img_np.shape[1], img_np.shape[2]), dtype=np.uint8)
        for i in range(img_np.shape[0]):
            img_gray[i] = cv2.cvtColor(img_np[i], cv2.COLOR_RGB2GRAY)

        enhanced_images = np.zeros_like(img_gray)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        for i in range(img_gray.shape[0]):
            enhanced_images[i] = clahe.apply(img_gray[i])

        edge_images = np.zeros_like(enhanced_images)
        for i in range(enhanced_images.shape[0]):
            edge_images[i] = cv2.Canny(enhanced_images[i], 20, 40)

        # 🔹 큐에 프레임 추가 (deque에 28개 저장)
        for i in range(64):
            frame_queues[i].append(edge_images[i])

        # 🔹 64개의 배치에 대해 5개 프레임 선택 (0, 7, 14, 21, 27번째)
        selected_indices = [0, 7, 14, 21, 27]
        stacked_frames = np.zeros((64, 5, edge_images.shape[1], edge_images.shape[2]), dtype=np.uint8)
        
        for i in range(64):
            frames_list = list(frame_queues[i])
            for j, idx in enumerate(selected_indices):
                if len(frames_list) > idx:  # deque가 충분한 길이를 가졌는지 확인
                    stacked_frames[i, j] = frames_list[idx]
                else:
                    stacked_frames[i, j] = frames_list[-1]  # 부족하면 최신 프레임을 사용

        # 🔹 터미널에 ASCII 출력 (🚀 최근 프레임(27번째)만 표시)
        os.system('clear' if os.name == 'posix' else 'cls') 
        print("\n🖥️ Contour Detection (ASCII View)\n" + "="*40)

        ascii_chars = ['.', '#']  
        resized = cv2.resize(stacked_frames[0, -1], (80, 40))  # 가장 최신 프레임 (27번째) 사용
        ascii_img = '\n'.join(
            ''.join(ascii_chars[1] if pixel > 0 else ascii_chars[0] for pixel in row)
            for row in resized
        )
        print(ascii_img)  

        # 🔹 PyTorch 텐서로 변환 (64, 5, height * width)
        contour_tensor = torch.from_numpy(stacked_frames).float() / 255.0
        contour_tensor = contour_tensor.to(images.device)
        reshaped_tensor = contour_tensor.view(64,-1)  

        return reshaped_tensor

    return images


def joint_vel_debug(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    print(f"left_joint_vel = {asset.data.joint_vel[0, 0]:.1f} , right_joint_vel = {asset.data.joint_vel[0, 4]:.1f} , {'왼쪽' if asset.data.joint_vel[0, 0] < asset.data.joint_vel[0, 4] else '오른쪽'}  {abs(asset.data.joint_vel[0, 0] - asset.data.joint_vel[0, 4]):.1f}  (Velocity Difference), Minus is go ahead, vel unit: rad/s, torque unit: Nm")# joint_vel(envs,asset_cfg.joint_ids)
    print(f"left_error = {abs(env.action_manager.action[0,0] - asset.data.joint_vel[0, 0]):.1f} , right_error = {abs(env.action_manager.action[0,1] - asset.data.joint_vel[0, 4]):.1f} ")
    print(f"torque:{asset.data.applied_torque[0, [0,1,4,5]]}")
    print(f"acc={asset.data.joint_acc[0, [0,1,4,5]]}")

    return asset.data.joint_vel[:, [0,1,4,5]]

    
def last_action_debug(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    if action_name is None:
        print(f"left_vel_input = {env.action_manager.action[0,0]:.1f} , right_vel_input = {env.action_manager.action[0,1]:.1f} rad/s, {'왼쪽' if env.action_manager.action[0,0] < env.action_manager.action[0,1] else '오른쪽'} = {abs(env.action_manager.action[0,0] - env.action_manager.action[0,1]):.1f} rad/s (Velocity Difference), Minus is go ahead, vel unit: rad/s, torque unit: Nm")
        return env.action_manager.action
    else:
        return env.action_manager.get_term(action_name).raw_actions
    


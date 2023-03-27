
import pickle
from refile import smart_open
from pyquaternion import Quaternion
import numpy as np
import os
import mmcv
import tqdm
sensors = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
# info_prefix = 'train'
# info_prefix = 'val'
info_prefix = 'test'
info_path = os.path.join("data/nuscenes/",'mmdet3d_nuscenes_30f_infos_{}.pkl'.format(info_prefix))
with smart_open('data/nuscenes/nuscenes_infos_{}.pkl'.format(info_prefix), "rb") as f:
    key_infos = pickle.load(f) ####nuscenes pkl
with smart_open('data/nuscenes/mmdet3d_key_nuscenes_12hz_infos_{}.pkl'.format(info_prefix), "rb") as f:
    key_sweep_infos = pickle.load(f) #### pkl contains previous key frames as sweep data, previous key frames has already aligned with current frame
with smart_open('data/nuscenes/nuscenes_12hz_infos_{}.pkl'.format(info_prefix), "rb") as f:
    sweep_infos = pickle.load(f) #### pkl contains origin sweep frames as sweep data, sweep frames has not aligned with current frame


num_prev = 5  ### previous key frame 
for current_id in tqdm.tqdm(range(len(sweep_infos))):
    ####current frame parameters
    e2g_t = key_infos['infos'][current_id]['ego2global_translation']
    e2g_r = key_infos['infos'][current_id]['ego2global_rotation']
    l2e_t = key_infos['infos'][current_id]['lidar2ego_translation']
    l2e_r = key_infos['infos'][current_id]['lidar2ego_rotation']
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix

    sweep_lists = []
    for i in range(num_prev):  #### previous key frame
        sample_id = current_id - i
        if sample_id < 0 or len(sweep_infos[sample_id]['sweeps']) == 0 or i >= len(key_sweep_infos['infos'][current_id]['sweeps']):
            continue
        for sweep_id in range(5): ###sweep frame for each previous key frame
            if len(sweep_infos[sample_id]['sweeps'][sweep_id].keys()) != 6:
                print(sample_id, sweep_id, sweep_infos[sample_id]['sweeps'][sweep_id].keys())
                temp = sweep_lists[-1]
                sweep_lists.append(temp)
                continue
            else:
                sweep_cams = dict()
                for view in sweep_infos[sample_id]['sweeps'][sweep_id].keys():
                    sweep_cam = dict()
                    sweep_cam['data_path'] = 'data/nuscenes'+ sweep_infos[sample_id]['sweeps'][sweep_id][view]['filename']
                    sweep_cam['type'] = 'camera'
                    sweep_cam['timestamp'] = sweep_infos[sample_id]['sweeps'][sweep_id][view]['timestamp']
                    sweep_cam['is_key_frame'] = sweep_infos[sample_id]['sweeps'][sweep_id][view]['is_key_frame']
                    sweep_cam['nori_id'] = sweep_infos[sample_id]['sweeps'][sweep_id][view]['nori_id']
                    sweep_cam['sample_data_token'] = sweep_infos[sample_id]['sweeps'][sweep_id][view]['sample_token']
                    sweep_cam['ego2global_translation']  = sweep_infos[sample_id]['sweeps'][sweep_id][view]['ego_pose']['translation']
                    sweep_cam['ego2global_rotation']  = sweep_infos[sample_id]['sweeps'][sweep_id][view]['ego_pose']['rotation']
                    sweep_cam['sensor2ego_translation']  = sweep_infos[sample_id]['sweeps'][sweep_id][view]['calibrated_sensor']['translation']
                    sweep_cam['sensor2ego_rotation']  = sweep_infos[sample_id]['sweeps'][sweep_id][view]['calibrated_sensor']['rotation']
                    sweep_cam['cam_intrinsic'] = sweep_infos[sample_id]['sweeps'][sweep_id][view]['calibrated_sensor']['camera_intrinsic']

                    l2e_r_s = sweep_cam['sensor2ego_rotation']
                    l2e_t_s = sweep_cam['sensor2ego_translation'] 
                    e2g_r_s = sweep_cam['ego2global_rotation']
                    e2g_t_s = sweep_cam['ego2global_translation'] 

                    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
                    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
                    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
                            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
                    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
                        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
                    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                                    ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
                    sweep_cam['sensor2lidar_rotation'] = R.T  # points @ R.T + T
                    sweep_cam['sensor2lidar_translation'] = T

                    lidar2cam_r = np.linalg.inv(sweep_cam['sensor2lidar_rotation'])
                    lidar2cam_t = sweep_cam['sensor2lidar_translation'] @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                    intrinsic = np.array(sweep_cam['cam_intrinsic'])
                    viewpad = np.eye(4)
                    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                    sweep_cam['intrinsics'] = viewpad
                    sweep_cam['extrinsics'] = lidar2cam_rt
                    sweep_cam['lidar2img'] = lidar2img_rt

                    pop_keys = ['ego2global_translation', 'ego2global_rotation', 'sensor2ego_translation', 'sensor2ego_rotation', 'cam_intrinsic']
                    [sweep_cam.pop(k) for k in pop_keys]
                    # sweep_cam= sweep_cam.pop(pop_keys)
                    sweep_cams[view] = sweep_cam
                sweep_lists.append(sweep_cams)
        ##key frame
        sweep_lists.append(key_sweep_infos['infos'][current_id]['sweeps'][i])
        ####suppose that previous key frame has aligned with current frame. The process of previous key frame is similar to the sweep frame before.
    key_infos['infos'][current_id]['sweeps'] = sweep_lists
mmcv.dump(key_infos, info_path)
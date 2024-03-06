import tensorflow as tf
import tensorflow_datasets as tfds
import argparse
import rlds
import h5py
import os
import glob

rlds_dataset_dir = '/home/qyb/RT-1/data/rt_dataset_builder/1.0.0/'
h5_dataset_dir = '/home/qyb/RT-1/data/h5_dataset_builder'
act_h5_dataset_dir = '/home/qyb/act/act/data/'
# data_dict = {
#     '/observations/qpos': [],
#     '/observations/qvel': [],
#     '/action': [],
# }
# for cam_name in camera_names:
#     data_dict[f'/observations/images/{cam_name}'] = []

# # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
# # truncate here to be consistent
# joint_traj = joint_traj[:-1]
# episode_replay = episode_replay[:-1]

# # len(joint_traj) i.e. actions: max_timesteps
# # len(episode_replay) i.e. time steps: max_timesteps + 1
# max_timesteps = len(joint_traj)
# while joint_traj:
#     action = joint_traj.pop(0)
#     ts = episode_replay.pop(0)
#     data_dict['/observations/qpos'].append(ts.observation['qpos'])
#     data_dict['/observations/qvel'].append(ts.observation['qvel'])
#     data_dict['/action'].append(action)
#     for cam_name in camera_names:
#         data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

# # HDF5
# t0 = time.time()
# dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
# with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
#     root.attrs['sim'] = True
#     obs = root.create_group('observations')
#     image = obs.create_group('images')
#     for cam_name in camera_names:
#         _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
#                                     chunks=(1, 480, 640, 3), )
#     # compression='gzip',compression_opts=2,)
#     # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
#     qpos = obs.create_dataset('qpos', (max_timesteps, 14))
#     qvel = obs.create_dataset('qvel', (max_timesteps, 14))
#     action = root.create_dataset('action', (max_timesteps, 14))

#     for name, array in data_dict.items():
#         root[name][...] = array
# print(f'Saving: {time.time() - t0:.1f} secs\n')


# def process_data(episode):
#     steps = episode['steps'].take(1).get_single_element()
#     action_lable = {
#         "terminate_episode": steps["action"]["terminate_episode"],
#         "rotation_delta": steps["action"]["rotation_delta"],
#         "gripper_closedness_action": steps["action"]["gripper_closedness_action"],
#         "world_vector": steps["action"]["world_vector"]
#     }

#     train_observation = {
#         "image": steps["observation"]["image"],
#         "natural_language_embedding": tf.cast(steps['observation']['natural_language_embedding'], tf.float32)
#     }

#     return {"action_lable": action_lable, "train_observation": train_observation}

def process_data(episode):  
    action_labels = {  
        "terminate_episode": [],  
        "rotation_delta": [],  
        "gripper_closedness_action": [],  
        "world_vector": []  
    }  
  
    train_observations = {  
        "image": [],  
        "natural_language_embedding": []  
    }  
  
    # 遍历所有steps  
    for step in episode['steps']:  
        for key, value in action_labels.items():  
            action_labels[key].append(step["action"][key])  
  
        train_observations["image"].append(step["observation"]["image"])  
        train_observations["natural_language_embedding"].append(tf.cast(step['observation']['natural_language_embedding'], tf.float32))  
  
    # 所有step的数据整合到一个字典中  
    all_steps_data = {  
        "action_labels": action_labels,  
        "train_observations": train_observations  
    }  
    return all_steps_data

def save_episode_to_h5(episode_dict, output_filename):
    with h5py.File(output_filename, 'w') as hf:
        for key, value in episode_dict.items():
            if isinstance(value, dict):
                group = hf.create_group(key)
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, tf.Tensor):
                        sub_value = sub_value.numpy()
                    group.create_dataset(sub_key, data=sub_value)
            else:
                if isinstance(value, tf.Tensor):
                    value = value.numpy()
                hf.create_dataset(key, data=value)

def read_h5_file(h5_file):
    # 打开HDF5文件  
    with h5py.File(h5_file, 'r') as f:  
        # 列出文件中的所有键（组或数据集）  
        print(list(f.keys()))  
    

def rlds2h5(rlds_dataset_dir, h5_dataset_dir):
    builder = tfds.builder_from_directory(builder_dir=rlds_dataset_dir)
    ds = builder.as_dataset(split='train')

    # 遍历数据集并逐个处理和保存
    for i,episode in enumerate(ds):
        processed_episode = process_data(episode)
        file_path = episode['episode_metadata']['file_path']
        path_str = file_path.numpy().decode('utf-8')  # 解码为字符串
        episode_id = os.path.basename(path_str)  # 提取最后一级目录或文件名
        print(episode_id)  # 输出：'episode47'
        output_filename = f"{h5_dataset_dir}/{episode_id}.hdf5"
        save_episode_to_h5(processed_episode, output_filename)

def main(args):
    # h5_dataset_dir = args['h5_dataset_dir']
    if not os.path.exists(h5_dataset_dir):
        os.makedirs(h5_dataset_dir)

    if not os.path.exists(rlds_dataset_dir):
        raise ValueError(f"The RLDS directory '{rlds_dataset_dir}' does not exist.")

    # rlds2h5(rlds_dataset_dir, h5_dataset_dir)
    h5_file =  os.path.join(act_h5_dataset_dir, 'episode_0.hdf5')  
    read_h5_file(h5_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--rlds_dataset_dir', action='store', type=str, help='rlds dataset dir', required=False, default='/home/qyb/RT-1/data/rt_dataset_builder/1.0.0/')
    #parser.add_argument('--h5_dataset_dir', action='store', type=str, help='h5 dataset dir', required=False, default='/home/qyb/RT-1/data/rt_dataset_builder/1.0.0/')

    main(vars(parser.parse_args()))
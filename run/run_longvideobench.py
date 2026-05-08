import sys
import os
try:
    from kmeans_pytorch import kmeans
    print("Successfully imported kmeans from kmeans_pytorch")
except ImportError:
    from kmeans_pytorch.kmeans_pytorch import kmeans
import torch
import json
import numpy as np
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
import argparse
import time
import pandas as pd
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
from all_model_agent_new import *
from all_model_util import *
import re
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import torch.multiprocessing as mp
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.modeling import CLIP4Clip

mp.set_start_method('spawn', force=True)
SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                 "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

model_path = 'path/pytorch_model0001.bin.44'
args = {
    'video_dim': 1024,
    'max_words': 60,
    'max_frames': 16,
    'feature_framerate': 1,
    'margin': 0.1,
    'hard_negative_rate': 0.5,
    'negative_weighting': 1,
    'n_pair': 1,
    'text_num_hidden_layers': 16,
    'visual_num_hidden_layers': 16,
    'cross_num_hidden_layers': 4,
    'linear_patch': "2d",
    'sim_header': "seqTransf"
}

def init_model(model_path, args, device):
    model_state_dict = torch.load(model_path, map_location='cpu')
    model = CLIP4Clip.from_pretrained("cross-base", cache_dir="", state_dict=model_state_dict, task_config=args)
    model.to(device)
    return model

def _get_text(tokenizer, video_id, sentence, device):
    choice_video_ids = [video_id]
    n_caption = len(choice_video_ids)
    k = n_caption
    pairs_text = np.zeros((k, 77), dtype=np.int64)
    pairs_mask = np.zeros((k, 77), dtype=np.int64)
    pairs_segment = np.zeros((k, 77), dtype=np.int64)

    words = tokenizer.tokenize(sentence)

    words = [SPECIAL_TOKEN["CLS_TOKEN"]] + words
    total_length_with_CLS = 76
    if len(words) > total_length_with_CLS:
        words = words[:total_length_with_CLS]
    words = words + [SPECIAL_TOKEN["SEP_TOKEN"]]

    input_ids = tokenizer.convert_tokens_to_ids(words)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    while len(input_ids) < 77:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    pairs_text[0] = np.array(input_ids)
    pairs_mask[0] = np.array(input_mask)
    pairs_segment[0] = np.array(segment_ids)
    pairs_text = torch.Tensor(pairs_text).to(device)
    pairs_mask = torch.Tensor(pairs_mask).to(device)
    pairs_segment = torch.Tensor(pairs_segment).to(device)
    return pairs_text.long(), pairs_mask.long(), pairs_segment.long(), choice_video_ids

@torch.no_grad()
def run_clip4clip(model, video_path, text, sample_idx, device):
    tokenizer = ClipTokenizer()
    input_ids, input_mask, segment_ids, choice_video_ids = _get_text(tokenizer, video_path, text, device)
    video = extract_and_resize_frames(video_path, sample_idx).to(device)
    video_mask = torch.Tensor([[1] * 16]).to(device)
    token_type_ids = torch.Tensor([[0] * 16]).to(device)
    visual_output = model.get_visual_output(video, video_mask=video_mask, shaped=True, video_frame=16)
    text_feat = model.get_sequence_output(input_ids, segment_ids, input_mask, shaped=True)
    b1b2_logits, *_tmp = model.get_similarity_logits(text_feat, visual_output, input_mask, video_mask,
                                                     loose_type=True, eval='myeval')
    return b1b2_logits


def extract_and_resize_frames(video_path, frame_indices):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    frames = vr.get_batch(frame_indices).asnumpy()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor()
    ])
    resized_frames = []
    for frame in frames:
        resized_frame = transform(frame)
        resized_frames.append(resized_frame)
    resized_frames = torch.stack(resized_frames)
    return resized_frames

def load_frame_features(name_ids, save_folder):
    filename = f"{name_ids}.pt"  
    filepath = os.path.join(save_folder, name_ids, filename)  
    if not os.path.exists(filepath) and name_ids.startswith("@"):
        match = re.search(r'(\d+)$', name_ids)
        if match:
            numeric_id = match.group(1)
            filename = f"{numeric_id}.pt"
            filepath = os.path.join(save_folder, numeric_id, filename)

    img_feats = torch.load(filepath, map_location='cpu')
    return img_feats

def find_furthest_split_frames(frame_feats, key_frames, cluster_ids_x, cluster_centers, device):
    shots = []
    shots.append({
        "keyframe_id": key_frames[0],
        "start_frame": 0,
        "end_frame": -1,
        "vis": 0
    })
    for i in range(len(key_frames) - 1):
        start_idx, end_idx = key_frames[i], key_frames[i + 1]
        if start_idx + 1 >= end_idx:
            shots[-1]["end_frame"] = start_idx
            shots.append({
                "keyframe_id": end_idx,
                "start_frame": end_idx,
                "end_frame": -1,
                "vis": 0
            })
            continue
        
        mid_indices = torch.arange(start_idx + 1, end_idx) 
        mid_frames = frame_feats[mid_indices] 
        cluster1 = cluster_centers[cluster_ids_x[start_idx]].to(device)  
        cluster2 = cluster_centers[cluster_ids_x[end_idx]].to(device)  

        dist1 = torch.norm(mid_frames - cluster1, dim=1)  
        dist2 = torch.norm(mid_frames - cluster2, dim=1)  
        total_dist = dist1 + dist2  
        furthest_idx = mid_indices[torch.argmax(total_dist)].item()
        shots[-1]["end_frame"] = furthest_idx - 1
        shots.append({
            "keyframe_id": key_frames[i + 1],
            "start_frame": furthest_idx,
            "end_frame": -1,
            "vis":0
        })
    shots[-1]["end_frame"] = len(frame_feats) - 1   
    for shot in shots:
        shot["num_frames"] = shot["end_frame"] - shot["start_frame"] + 1
    return shots

def first_cluster(frame_feats, device):
    cluster_num = 6  
    cluster_num = min(cluster_num, frame_feats.shape[0])
    cluster_ids_x, cluster_centers = kmeans(
        X=frame_feats, 
        num_clusters=cluster_num, 
        distance='cosine', 
        device=device
    )

    cluster_ids_x = cluster_ids_x.to(device)
    cluster_centers = cluster_centers.to(device)
    closest_points_idx_per_cluster = find_closest_points_per_cluster(frame_feats, cluster_ids_x, cluster_centers)
    key_frames = sorted([value for sublist in closest_points_idx_per_cluster.values() for value in sublist])
    shots = find_furthest_split_frames(frame_feats, key_frames, cluster_ids_x, cluster_centers, device)
    return shots, cluster_ids_x, cluster_centers


def find_closest_points_per_cluster(x, cluster_ids, cluster_centers):
    closest_points_idx_per_cluster = {cluster_id: [] for cluster_id in range(len(cluster_centers))}
    for cluster_id in range(len(cluster_centers)):
        indices_in_cluster = torch.where(cluster_ids == cluster_id)[0] 
        points_in_cluster = x[indices_in_cluster]  
        distances = torch.norm(points_in_cluster - cluster_centers[cluster_id], dim=1) 
        if distances.numel() > 0: 
            closest_idx_in_cluster = torch.argmin(distances).item()
            closest_global_idx = indices_in_cluster[closest_idx_in_cluster].item() 
            closest_points_idx_per_cluster[cluster_id].append(closest_global_idx) 

    return closest_points_idx_per_cluster 

def update_shot_frame_numbers(shots, all_image_paths):
    transfer_shots = []
    for shot in shots:
        new_shot = shot.copy()

        start_image_path = all_image_paths[shot["start_frame"]]
        start_frame_number = int(re.search(r'(\d+)\.jpg$', start_image_path).group(1))
        new_shot["start_frame"] = start_frame_number
        
        end_image_path = all_image_paths[shot["end_frame"]]
        end_frame_number = int(re.search(r'(\d+)\.jpg$', end_image_path).group(1))
        new_shot["end_frame"] = end_frame_number
        
        if "keyframe_id" in shot:
            keyframe_image_path = all_image_paths[shot["keyframe_id"]]
            keyframe_number = int(re.search(r'(\d+)\.jpg$', keyframe_image_path).group(1))
            new_shot["keyframe_id"] = keyframe_number
        
        transfer_shots.append(new_shot)
    return transfer_shots

def get_max_frame_shot(video_path, transfer_shots, text_prompt, asp_clip, device, clip_sample_frame=16, qwen_sample_frame=32, top_n=1):
    shot_scores = []  
    for idx, shot in enumerate(transfer_shots):
        qwen_frame_idx = get_frame_idx_path(video_path, shot, qwen_sample_frame)
        
        if len(qwen_frame_idx) >= clip_sample_frame:
            step = len(qwen_frame_idx) / clip_sample_frame
            clip_frame_idx = [qwen_frame_idx[int(i * step)] for i in range(clip_sample_frame)]
        else:
            clip_frame_idx = qwen_frame_idx
            
        clip_score = run_clip4clip(asp_clip, video_path, text_prompt, clip_frame_idx, device)
        shot_scores.append({
            "index": idx,      
            "shot": shot,
            "clip_score": clip_score,
            "frame_idx": qwen_frame_idx
        })

    shot_scores.sort(key=lambda x: x["clip_score"], reverse=True)

    if top_n == 1:
        best = shot_scores[0]
        return best["frame_idx"], best["shot"], best["index"]
    else:
        top_n = min(top_n, len(shot_scores))
        best_sampled_idx_list = [shot_scores[i]["frame_idx"] for i in range(top_n)]
        select_shot_list = [shot_scores[i]["shot"] for i in range(top_n)]
        select_shot_idx_list = [shot_scores[i]["index"] for i in range(top_n)]
        return best_sampled_idx_list, select_shot_list, select_shot_idx_list

def get_frame_idx_path(video_path, shot = None, sample_frame=16, judge_whole=False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    if len(vr) <= 96 or judge_whole:
        sorted_frame_idx = random.sample(range(len(vr)), sample_frame)
        sorted_frame_idx = sorted(sorted_frame_idx)
        return sorted_frame_idx 
    start = shot["start_frame"]
    end = shot["end_frame"]
    try:
        sorted_frame_idx = random.sample(range(start, end + 1), sample_frame)
    except:
        sorted_frame_idx = random.sample(range(len(vr)), sample_frame)

    sorted_frame_idx = sorted(sorted_frame_idx)
    return sorted_frame_idx

def sample_additional_frames_from_diff(video_path, diff, sampled_idx, sample_frame=16):
    new_samples = []
    
    for shot in diff:
        start = shot["start_frame"]
        end = shot["end_frame"]
        all_frames = list(range(start, end + 1))
        candidate_frames = [f for f in all_frames if f not in sampled_idx]

        if not candidate_frames:
            continue
        if len(candidate_frames) < sample_frame:
            sampled = candidate_frames
        else:
            sampled = random.sample(candidate_frames, sample_frame)
        new_samples.extend(sampled)
    
    updated_sampled_idx = sorted(list(set(sampled_idx) | set(new_samples)))
    return updated_sampled_idx

def expansion_round(item, frame_feats, shots, cluster_ids_x, cluster_centers, expansion_shots, device):
    new_cluster_centers = []   
    new_cluster_ids_x = torch.empty_like(cluster_ids_x)  
    updated_shots = []  
    
    cluster_counter = 0  
    for shot_idx, shot in enumerate(shots):
        start = shot["start_frame"]
        end = shot["end_frame"]
        shot_frames = frame_feats[start:end+1]
        num_frames = shot_frames.shape[0]
        if num_frames <= 0:
            continue

        if shot_idx not in expansion_shots:
            orig_cluster_id = int(cluster_ids_x[start].item())
            new_center = cluster_centers[orig_cluster_id].cpu()
            new_cluster_centers.append(new_center)
            for i in range(start, end+1):
                new_cluster_ids_x[i] = cluster_counter

            updated_shots.append(shot)
            cluster_counter += 1
        else:
            if num_frames == 1:
                new_cluster_ids_x[start] = cluster_counter
                new_cluster_centers.append(shot_frames[0].cpu())
                updated_shots.append(shot)
                cluster_counter += 1
                continue

            k_shot = min(2, num_frames)
            shot_cluster_ids, shot_cluster_centers = kmeans(
                X=shot_frames, num_clusters=k_shot, distance='cosine',device=device
            )
            shot_cluster_ids = shot_cluster_ids.to(device)
            shot_cluster_centers = shot_cluster_centers.to(device)

            for local_idx in range(num_frames):
                global_idx = start + local_idx
                new_cluster_ids_x[global_idx] = cluster_counter + int(shot_cluster_ids[local_idx].item())

            for j in range(k_shot):
                new_cluster_centers.append(shot_cluster_centers[j].cpu())
            local_cluster_start = cluster_counter
            cluster_counter += k_shot

            shot_closest_points = find_closest_points_per_cluster(shot_frames, shot_cluster_ids, shot_cluster_centers)
            local_key_frames = []
            for cid, local_indices in shot_closest_points.items():
                if local_indices:
                    local_kf = local_indices[0] 
                    local_key_frames.append(local_kf)
                    global_kf = start + local_kf
            local_key_frames.sort()

            local_shots = find_furthest_split_frames(shot_frames, local_key_frames, shot_cluster_ids, shot_cluster_centers, device)
            for ls in local_shots:
                ls['keyframe_id'] += start
                ls['start_frame'] += start
                ls['end_frame'] += start
                updated_shots.append(ls)
    
    new_cluster_centers_tensor = torch.stack(new_cluster_centers) if new_cluster_centers else torch.tensor([])
    return updated_shots, new_cluster_ids_x, new_cluster_centers_tensor

def load_or_compute_cluster_results(video_id, frame_feat_path, all_image_paths, device='cuda'):
    save_folder = "yourpath/first_time_cluster/lvbench"
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"{video_id}.pt")
    
    if os.path.exists(save_path):
        data = torch.load(save_path, map_location=device)
        return data["shots"], data["transfer_shots"], data["cluster_ids_x"], data["cluster_centers"]
    else:
        frame_feats = load_frame_features(video_id, frame_feat_path).to(device)
        shots, cluster_ids_x, cluster_centers = first_cluster(frame_feats, device)
        transfer_shots = update_shot_frame_numbers(shots, all_image_paths)
        
        data_to_save = {
            "shots": shots,
            "transfer_shots": transfer_shots,
            "cluster_ids_x": cluster_ids_x,
            "cluster_centers": cluster_centers,
        }
        torch.save(data_to_save, save_path)
        return shots, transfer_shots, cluster_ids_x, cluster_centers

def normalize_frame_folder(frame_folder: str) -> str:
    if not os.path.isdir(frame_folder):
        base_dir, folder_name = os.path.split(frame_folder)
        if folder_name.startswith('@'):
            m = re.search(r'(\d+)$', folder_name)
            if m:
                numeric_id = m.group(1)
                candidate = os.path.join(base_dir, numeric_id)
                if os.path.isdir(candidate):
                    return candidate
    return frame_folder


def normalize_video_path(video_path: str) -> str:
    if not os.path.exists(video_path):
        dirname, fname = os.path.split(video_path)
        if fname.startswith('@'):
            m = re.search(r'(\d+)\.mp4$', fname)
            if m:
                candidate = os.path.join(dirname, f"{m.group(1)}.mp4")
                if os.path.exists(candidate):
                    return candidate
    return video_path

def calc_one_question(model, asp_clip, item, idx, device):
    video_id = item["video_id"]
    item["ans"] = chr(ord('A') + item['correct_choice'])
    frame_folder = f"yourpath/longvideobench/frames/videos/{video_id}"
    video_path = os.path.join('yourpath/LongVideoBench/videos', video_id + '.mp4')
    frame_feat_path = "yourpath/longvideobench/features/videos"
    frame_feats = load_frame_features(video_id, frame_feat_path).to(device)
    frame_folder = normalize_frame_folder(frame_folder)
    video_path = normalize_video_path(video_path)
    all_image_paths = sorted(
        [
            os.path.join(frame_folder, fname)
            for fname in os.listdir(frame_folder)
            if fname.endswith('.jpg')
        ],
        key=lambda x: int(re.search(r'(\d+)\.jpg$', x).group(1))
    )
    vote_options = [] 
    num_round = 0
    prompt_watch, prompt_info, prompt_question = get_lvbench_prompt(item)
    item["shot"] = []
    model_name = "qwen7b"
    sampled_idx = get_frame_idx_path(video_path, shot = None, sample_frame=4, judge_whole=True)

    item["watch"] = model.get_answer(video_path, prompt_watch, sampled_idx)[0]
    print("watch: ", item["watch"], ('Yes' in item["watch"]))

    if 'Yes' in item["watch"] or 'Y' in item["watch"]:
        sampled_idx = get_frame_idx_path(video_path, shot = None, sample_frame=32, judge_whole=True)
        res = model.get_answer(video_path, prompt_question, sampled_idx)
        
        item["qwen7b"] = res[0].split('Answer: ')[-1][0]
        if item["qwen7b"] == item['ans']:
            item["final_sampled_idx"] = sampled_idx 
            item["num_round"] = num_round
            vote_options.append(item["qwen7b"])
            return item, vote_options
        item["num_round"] = num_round
        item["final_sampled_idx"] = sampled_idx 
        vote_options.append(item["qwen7b"])
        return item, vote_options
    vote_options.append(-1) 

    num_round = 1 
    shots, transfer_shots, cluster_ids_x, cluster_centers = load_or_compute_cluster_results(
        video_id, frame_feat_path, all_image_paths, device=device
    )
    item["round1"] = {
        "shots": [],    
        "correct_shots": []  
    }
    key_info = ""
    if "info" not in item["round1"]: 
        res = model.get_answer(video_path, prompt_info, sampled_idx)
        res = (res[0] if isinstance(res, list) and res else res)
        key_info = item["round1"]["info"] = res
    
    sampled_idx, _, _ = get_max_frame_shot(video_path, transfer_shots, key_info, asp_clip, device, clip_sample_frame=16, qwen_sample_frame=32, top_n=1)    
   
    res = model.get_answer(video_path, prompt_question, sampled_idx)

    item["qwen7b"] = item["round1"]["res"] = res[0].split('Answer: ')[-1][0]

    reflect_prompt = lvbench_reflection(item, num_round)
    reflect_res = model.get_answer(video_path, reflect_prompt, sampled_idx)
    reflect_res = (reflect_res[0] if isinstance(reflect_res, list) and reflect_res else reflect_res)
    item["round1"]["reason"] = reflect_res

    now_round = "round" + str(num_round)
    
    prompt = lvbench_check_if_sufficient(item, now_round, model_name)
    ans_check = model.get_answer(video_path, prompt, sampled_idx)

    s = str(ans_check)
    match = re.search(r'\d+', s)
    confidence = int(match.group()) if match else 0
   
    if confidence == 3:
        item["final_sampled_idx"] = sampled_idx
        item["num_round"] = num_round
        vote_options.append(item["qwen7b"])
        return item, vote_options
    vote_options.append(item["qwen7b"])

    for i in range(5):  
        print(f"----  {i+1} round ----")
        num_round += 1
        now_round = "round" + str(num_round)
        item[now_round] = { "shots": [], "correct_shots": []  }

        _, _, expansion_shots = get_max_frame_shot(video_path, transfer_shots, key_info, asp_clip, device, clip_sample_frame=16, qwen_sample_frame=32, top_n=2)
        expansion_shots = sorted(expansion_shots)

        shots, cluster_ids_x, cluster_centers = expansion_round(item, frame_feats, shots,  cluster_ids_x, cluster_centers, expansion_shots, device)
        tmp_shots = transfer_shots.copy()
        transfer_shots = update_shot_frame_numbers(shots, all_image_paths)
        diff_shots = []
        for shot in transfer_shots:
            if shot not in tmp_shots:
                diff_shots.append(shot)
        sampled_idx = sample_additional_frames_from_diff(video_path, diff_shots, sampled_idx, sample_frame=8)

        updata_info_prompt = lvbench_update_info(item, num_round)

        res = model.get_answer(video_path, updata_info_prompt, sampled_idx)
        res = (res[0] if isinstance(res, list) and res else res)
        key_info = item[now_round]["info"] = res

        num_frames = len(sampled_idx)
        res = model.get_answer(video_path, prompt_question, sampled_idx)
        item["qwen7b"] = item[now_round]["res"] = res[0].split('Answer: ')[-1][0]
        vote_options.append(item["qwen7b"])
        
        reflect_prompt = lvbench_reflection(item, num_round)
        reflect_res = model.get_answer(video_path, reflect_prompt, sampled_idx)
        reflect_res = (reflect_res[0] if isinstance(reflect_res, list) and reflect_res else reflect_res)
        item[now_round]["reason"] = reflect_res

        prompt = lvbench_check_if_sufficient(item, now_round, model_name)
        ans_check = model.get_answer(video_path, prompt, sampled_idx)
        s = str(ans_check)
        match = re.search(r'\d+', s)
        confidence = int(match.group()) if match else 0
        if confidence == 3:
            item["final_sampled_idx"] = sampled_idx
            item["num_round"] = num_round
            return item, vote_options

    num_round += 1
    res_vote = vote_options
    if vote_options:
        vote_counter = Counter(vote_options)
        final_vote_option = vote_counter.most_common(1)[0][0]
        item["qwen7b"] = final_vote_option
        item["final_sampled_idx"] = sampled_idx
        item["num_round"] = num_round
        
    return item, res_vote

def worker(rank, world_size, all_anns, shared_results, mme_json, model_path, args, lock):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    local_cnt = 0 
    model_agent = Qwen2_5_7bAgent(device=device)
    local_asp_clip = init_model(model_path, args, device) 

    indices = [i for i in range(len(all_anns)) if i % world_size == rank]
    
    for idx in indices:
        item = all_anns[idx]
        if "qwen7b" in item:
            if rank == 0: print(f"Skipping index {idx}, already processed.")
            continue
        
        try:
            item, vote = calc_one_question(model_agent, local_asp_clip, item, idx, device) 
            shared_results.append({
                "index": idx,
                "qwen7b": item["qwen7b"],
                "ans": item["ans"],
                "num_round": item["num_round"]
            })
            local_cnt += 1
            if len(shared_results) % 5 == 0:
                with lock:
                    save_shared_data(all_anns, shared_results, mme_json)
                if rank == 0:
                    print_global_status(shared_results, len(shared_results))
                    
        except Exception as e:
            print(f"Process {rank} error: {e}")


def print_global_status(shared_results, total_done):
    results_list = list(shared_results)
    right = sum(1 for x in results_list if x["qwen7b"] == x["ans"])
    acc = right / total_done
    print("\n" + "="*30)
    print(f"🔥 已完成: {total_done})")
    print(f"✅ 正确数: {right} | ❌ 错误数: {total_done - right}")
    print(f"📈 当前总准确率: {acc:.2%}")
    print("="*30 + "\n")


def save_shared_data(all_anns, shared_results, mme_json):
    for res in shared_results:
        idx = res["index"]
        all_anns[idx]["qwen7b"] = res["qwen7b"]
    with open(mme_json, "w") as f:
        json.dump(all_anns, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    mme_json = "yourpath/lvbench.json"
    anns = json.load(open(mme_json, "r"))
    num_gpus = 8
    manager = mp.Manager()
    shared_results = manager.list() 
    
    lock = manager.Lock() 
    mp.spawn(worker, nprocs=num_gpus, args=(num_gpus, anns, shared_results, mme_json, model_path, args, lock))
    save_shared_data(anns, shared_results, mme_json)
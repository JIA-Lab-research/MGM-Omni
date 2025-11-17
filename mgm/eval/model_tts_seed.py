import argparse
import os
import json
from tqdm import tqdm
import json
import string
import re
import math
import numpy as np
from tqdm import tqdm
import random
import datetime
import accelerate

from mgm.constants import DEFAULT_SPEECH_TOKEN, AUDIO_START, AUDIO_END
from mgm.conversation import conv_templates
from mgm.model.builder import load_pretrained_model
from mgm.utils import disable_torch_init
from mgm.mm_utils import tokenizer_speech_token

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import librosa
import soundfile as sf
import torchaudio
from torchaudio.transforms import Resample


def dump_speech(args, pred_wav, file_name):
    sf.write(f"{args.out_speech_path}/{file_name}", pred_wav, 24000)


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class CustomDataset(Dataset):
    def __init__(
        self, speech_test_file, tokenizer, device,
        num_chunks, chunk_idx, speech_folder,
        conv_mode='qwen2vl', language='en'):

        self.speech_test_file = speech_test_file
        self.tokenizer = tokenizer
        self.target_sample_rate = 24000
        self.speech_folder = speech_folder
        self.conv_mode = conv_mode
        self.device = device

        pre_prompt_en = 'Respond with the tone of the reference audio clip.'
        pre_prompt_cn = '使用提供的音频片段的语气回复。'
        self.pre_prompt = (pre_prompt_en if language == 'en' else pre_prompt_cn)

        post_prompt_en = 'You may start speaking.'
        post_prompt_cn = '现在开始说话。'
        self.post_prompt = (post_prompt_en if language == 'en' else post_prompt_cn)
        
        test_data = []
        with open(speech_test_file, 'r') as f:
            for i, line in enumerate(f):
                id, ref_text, ref_audio, pred_text = line.strip().split('|')
                ref_audio = os.path.join(self.speech_folder, ref_audio)
                if language == 'en':
                    pred_text = ' '.join(pred_text.split(' '))
                else:
                    pred_text = ''.join(pred_text.split(' '))
                item = dict(id=id, ref_text=ref_text, ref_audio=ref_audio, pred_text=pred_text)
                test_data.append(item)
                
        self.test_data = get_chunk(test_data, num_chunks, chunk_idx)

    def __getitem__(self, index):
        ref_text = self.test_data[index]["ref_text"]
        pred_text = self.test_data[index]["pred_text"]
        ref_audio = self.test_data[index]["ref_audio"]
        file_name = self.test_data[index]["id"]+'.wav'

        qs = self.pre_prompt + AUDIO_START + DEFAULT_SPEECH_TOKEN + AUDIO_END + '\n' # + self.post_prompt
        answer = AUDIO_START + pred_text

        conv = conv_templates['qwen2vl'].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], answer)
        prompt = conv.get_prompt()
        input_ids = tokenizer_speech_token(prompt, self.tokenizer, return_tensors='pt').to(self.device)

        input_ids_refer = self.tokenizer(ref_text)['input_ids']
        input_ids_refer = torch.tensor(input_ids_refer).to(self.device)
        audio_refer, _ = librosa.load(ref_audio, sr=16000)
        audio_refer = torch.tensor(audio_refer).to(self.device)

        return input_ids, input_ids_refer, audio_refer, file_name
    
    def __len__(self):
        return len(self.test_data)


def create_data_loader(
    speech_test_file, tokenizer, device,
    speech_folder, conv_mode, language,
    batch_size=1, num_workers=4, num_chunks=4, chunk_idx=0):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(
        speech_test_file, tokenizer, device,
        num_chunks, chunk_idx, speech_folder, conv_mode, language)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # build model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    tokenizer, model = load_pretrained_model(model_path, cosyvoice_path=args.cosyvoice_path)

    if not os.path.exists(args.out_speech_path):
        os.makedirs(args.out_speech_path, exist_ok=True)
    
    # build dataloader
    data_loader = create_data_loader(args.speech_test_file, tokenizer, model.device,
                                     args.speech_folder, args.conv_mode, args.language,
                                     num_workers=args.num_workers, 
                                     num_chunks=args.num_chunks, chunk_idx=args.chunk_idx)
    
    # inference
    for input_ids, input_ids_refer, audio_refer, file_name in tqdm(data_loader, total=len(data_loader)):
        output_path = os.path.join(args.out_speech_path, file_name[0])
        if os.path.exists(output_path):
            print('Exists, skip', file_name[0])
            continue

        with torch.inference_mode():
            temperature = args.temperature
            for i in range(5):
                speech_ids, audio, tts_error = model.generate(
                    input_ids.clone(),
                    input_ids_refer=input_ids_refer.clone(),
                    audio_refer=audio_refer,
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    max_new_tokens=args.max_new_tokens,
                    bos_token_id=tokenizer.pad_token_id,
                    eos_token_id=[tokenizer.eos_token_id],
                    pad_token_id=tokenizer.pad_token_id,
                    tokenizer=tokenizer,
                    check_tts_result=True,
                    use_cache=True)
                if tts_error == False:
                    if i > 0:
                        print('Trial', i, 'Success', file_name[0], speech_ids.shape)
                    break
                print('Trial', i, 'Fail', file_name[0], speech_ids.shape)
                temperature = max(1.0, temperature + 0.1)
        dump_speech(args, audio, file_name[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="wcy1122/MGM-Omni-TTS-2B")
    parser.add_argument("--cosyvoice-path", type=str, default=None)
    parser.add_argument("--out-speech-path", type=str, default="outputs")
    parser.add_argument("--speech-test-file", type=str, default="")
    parser.add_argument("--speech-folder", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="qwen2vl")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    args = parser.parse_args()

    eval_model(args)

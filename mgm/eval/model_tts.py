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


def dump_speech(args, pred_wav, output_path):
    sf.write(output_path, pred_wav, 24000)


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
        conv_mode='qwen2vl', language=None):

        self.speech_test_file = speech_test_file
        self.tokenizer = tokenizer
        self.target_sample_rate = 24000
        self.speech_folder = speech_folder
        self.conv_mode = conv_mode
        self.device = device
        self.pre_prompt_en = 'Respond with the tone of the reference audio clip.'
        self.pre_prompt_zh = '使用提供的音频片段的语气回复。'

        self.post_prompt_en = 'You may start speaking.'
        self.post_prompt_zh = '现在开始说话。'

        ref_audio_en = '/home/vnmember05/research/MGM-Omni-dev/assets/ref_audio/Man_EN.wav'
        ref_audio_en, _ = librosa.load(ref_audio_en, sr=16000)
        self.ref_audio_en = torch.tensor(ref_audio_en).to(device)
        self.ref_audio_text_en = '\"Incredible!\" Dr. Chen exclaimed, unable to contain her enthusiasm. \"The quantum fluctuations we have observed in these superconducting materials exhibit completely unexpected characteristics.\"'
        input_ids_refer = tokenizer(self.ref_audio_text_en)['input_ids']
        self.ref_input_ids_en = torch.tensor(input_ids_refer).to(device)

        ref_audio_zh = '/home/vnmember05/research/MGM-Omni-dev/assets/ref_audio/Man_ZH.wav'
        ref_audio_zh, _ = librosa.load(ref_audio_zh, sr=16000)
        self.ref_audio_zh = torch.tensor(ref_audio_zh).to(device)
        self.ref_audio_text_zh = '他疯狂寻找到能够让自己升级的办法终于有所收获，那就是炼体。'
        input_ids_refer = tokenizer(self.ref_audio_text_zh)['input_ids']
        self.ref_input_ids_zh = torch.tensor(input_ids_refer).to(device)

        test_data = []
        if '.jsonl' in speech_test_file:
            with open(speech_test_file, 'r', encoding='utf-8') as f:
                for line in f:
                    sample = json.loads(line)
                    item = dict(id=sample['id'], pred_text=sample['text'], language=sample['language'])
                    if language is None or language == sample['language']:
                        test_data.append(item)
        elif '.json' in speech_test_file:
            samples = json.load(open(speech_test_file, 'r', encoding='utf-8'))
            for sample in samples:
                item = dict(id=sample['id'], pred_text=sample['text'], language=sample['language'])
                if language is None or language == sample['language']:
                    test_data.append(item)
        else:
            raise NotImplementedError
        self.test_data = get_chunk(test_data, num_chunks, chunk_idx)

    def __getitem__(self, index):
        pred_text = self.test_data[index]["pred_text"]
        file_name = self.test_data[index]["id"] + '.wav'
        language = self.test_data[index]["language"]

        if language == 'en':
            pre_prompt = self.pre_prompt_en
            post_prompt = self.post_prompt_en
            ref_input_ids = self.ref_input_ids_en.clone()
            ref_audio = self.ref_audio_en
        else:
            pre_prompt = self.pre_prompt_zh
            post_prompt = self.post_prompt_zh
            ref_input_ids = self.ref_input_ids_zh.clone()
            ref_audio = self.ref_audio_zh

        qs = pre_prompt + AUDIO_START + DEFAULT_SPEECH_TOKEN + AUDIO_END + '\n' # + post_prompt
        answer = AUDIO_START + pred_text

        conv = conv_templates['qwen2vl'].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], answer)
        prompt = conv.get_prompt()
        input_ids = tokenizer_speech_token(prompt, self.tokenizer, return_tensors='pt').to(self.device)

        return input_ids, ref_input_ids, ref_audio, file_name
    
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
        dump_speech(args, audio, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="wcy1122/MGM-Omni-TTS-2B")
    parser.add_argument("--cosyvoice-path", type=str, default=None)
    parser.add_argument("--out-speech-path", type=str, default="outputs")
    parser.add_argument("--speech-test-file", type=str, default="")
    parser.add_argument("--speech-folder", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="qwen2vl")
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    args = parser.parse_args()

    eval_model(args)

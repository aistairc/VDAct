import sys
import math
from datetime import timedelta

import pandas as pd

import shutil
import warnings
import evaluate
from datasets import load_dataset
from torch.utils.data import DataLoader
from lightning.pytorch import seed_everything

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from videollava.model import LlavaLlamaForCausalLM, LlavaMPTForCausalLM
from videollava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, \
    DEFAULT_VIDEO_PATCH_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN

from logger import Logger
from config import ArgsManager
from utils import *


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def load_pretrained_model(model_path, model_base, model_name, lora_checkpoint=None, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            # path with adapter_model.bin
            model = PeftModel.from_pretrained(model, lora_checkpoint if lora_checkpoint is not None else model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    # ==========================================================================================================
    processor = {'image': None, 'video': None}

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        if model.config.mm_image_tower is not None:
            image_tower = model.get_image_tower()
            if not image_tower.is_loaded:
                image_tower.load_model()
            image_tower.to(device=device, dtype=torch.float16)
            image_processor = image_tower.image_processor
            processor['image'] = image_processor

        if model.config.mm_video_tower is not None:
            video_tower = model.get_video_tower()
            if not video_tower.is_loaded:
                video_tower.load_model()
            video_tower.to(device=device, dtype=torch.float16)
            video_processor = video_tower.video_processor
            processor['video'] = video_processor
    # ==========================================================================================================

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, processor, context_len


def initialize_model():
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name, lora_checkpoint=args.lora_checkpoint,
        load_8bit=args.load_8bit, load_4bit=args.load_4bit, device=args.global_rank,
        cache_dir=args.cache_dir)
    video_processor = processor["video"]
    return tokenizer, model, processor, video_processor


def inference():
    tokenizer, model, processor, video_processor = initialize_model()
    model.eval()

    # load dataset
    dataset = load_dataset("json", data_files=f"{args.data_dir}/{args.data_file}")["train"]
    all_dial_id_list = open(f"{args.data_dir}/{data_mapping[args.data_set]}").read().splitlines()
    dial_id_list = get_chunk(all_dial_id_list, args.num_chunks, args.chunk_idx)
    dial_id_dict = {dial_id: 1 for dial_id in dial_id_list}
    # dial_id_dict = read_dialogue_ids(f"{args.data_dir}/{data_mapping[args.data_set]}")
    dataset = dataset.filter(lambda x: dial_id_dict.get(x["id"]))
    if args.low_resource:
        dataset = dataset.train_test_split(test_size=0.01, shuffle=False)["test"]

    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=collator,
        shuffle=False,
        batch_size=1
    )

    logger.log_info(f"process rank#{args.global_rank}: "
                    f"inference on {len(dataloader)} out of {len(dataset)} dialogues "
                    f"(low resource={args.low_resource})")

    model.get_video_tower().config.num_frames = args.num_video_frames

    results = []
    for i, data in enumerate(dataloader):
        video_sum = ""
        if args.include_summary:
            act_sum = []
            for act_vid_file in data["activity_videos"]:
                act_vid_chunks = act_vid_file.split("/")
                act_sum_file = act_vid_chunks[-1].split("_")[0].replace(" ", "_") + f"_{act_vid_chunks[3]}.txt"
                act_sum.extend(open(f"{args.data_dir}/video_summaries/{act_sum_file}", "r").read().splitlines())
            video_sum = ("The following is a sequence of activities in the video from start to finish: " +
                         " ".join(act_sum))
            sum_enc = tokenizer.encode(video_sum)
            video_sum = tokenizer.decode(sum_enc[:1024]) if len(sum_enc) > 1024 else video_sum
        video_file = f"{args.resource_dir}/{data['scenario_video']}"
        video_tensor = video_processor(video_file, return_tensors='pt')['pixel_values']
        if type(video_tensor) is list:
            tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
        else:
            tensor = video_tensor.to(model.device, dtype=torch.float16)

        conv = conv_templates[args.conv_mode].copy()
        roles = conv.roles

        # logger.log_info_on_main_process(args.global_rank, f"SCENARIO#{i + 1}\n")

        for j, turn in enumerate(data["turns"]):
            inp = turn["question"]
            # logger.log_info_on_main_process(args.global_rank, f"{roles[0]}: {inp}")

            if j == 0:
                inp = (video_sum + ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) +
                       "\n" + inp)

            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(args.global_rank)

            current_max_len = input_ids.shape[1] + args.max_new_tokens
            if current_max_len - args.max_input_length > 0:
                print('Warning: The number of tokens in current conversation exceeds the max length. '
                      'The model will not see the contexts outside the range.')
            begin_idx = max(0, current_max_len - args.max_input_length)
            input_ids = input_ids[:, begin_idx:]

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            # logger.log_info_on_main_process(args.global_rank, f"Prompt:\n---\n{prompt}\n---\n")

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=tensor,
                    do_sample=args.do_sample,
                    top_p=args.top_p,
                    min_length=args.min_length,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            pred = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            if pred.endswith(stop_str):
                pred = pred[:-len(stop_str)]
            pred = pred.strip()

            # logger.log_info_on_main_process(args.global_rank, f"{roles[1]}: {pred} (GT: {turn['answer']})")

            conv.messages[-1][1] = turn["answer"]
            results.append({"id": f'{data["id"]}{j+1:02d}',
                            "dial_id": data["id"],
                            "turn_num": j+1,
                            "question": turn["question"],
                            "ref_answer": turn["answer"],
                            "gen_answer": pred})

        if i % args.logging_every == 0:
            logger.log_info(f"process rank#{args.global_rank}: {i / len(dataloader) * 100:.2f}% completed")

    logger.log_info(f"process rank#{args.global_rank}: all done")

    res_df = pd.DataFrame.from_records(results)

    # bleu_metric = evaluate.load("bleu")
    # refs = res_df["ref_answer"].tolist()
    # preds = res_df["gen_answer"].tolist()
    # eval_res = bleu_metric.compute(references=refs, predictions=preds)
    # logger.log_info(f"evaluation on {len(results)} turns: {eval_res}")

    res_df.to_json(args.res_file, orient="records", indent=2)
    logger.log_info(f"precess rank #{args.global_rank}: "
                    f"saved results of {len(res_df['dial_id'].unique())} dials "
                    f"({len(res_df)} turns) to {args.res_file}")


if __name__ == '__main__':
    args_manager = ArgsManager()
    args_manager.add_inference_args()
    parser = args_manager.get_args()
    parser.add_argument("--model-path", default="LanguageBind/Video-LLaVA-7B")
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--lora-checkpoint", default=None)
    parser.add_argument("--cache-dir", default="Video-LLaVA/cache_dir")
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=8)
    parser.add_argument("--res-file", default="", required=True)
    parser.add_argument("--include-summary", default=False, action="store_true")

    # for lora,
    # model_base=LanguageBind/Video-LLaVA-7B,
    # model_path=./checkpoints/videollava-7b-lora-8f/
    # lora_checkpoint=./checkpoints/videollava-7b-lora-8f/checkpoint-18

    args = parser.parse_args()
    args.global_rank = args.chunk_idx
    args.data_dir = os.path.join(args.resource_dir, "data")
    # args = args_manager.set_args(args)
    args.model_trademark_name = 'videollava'
    print(f"process rank#{args.global_rank}, {args}")

    seed_everything(args.seed)

    logger = Logger(program=sys.argv[0])

    disable_torch_init()
    inference()

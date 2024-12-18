import sys
import math

import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig, PretrainedConfig, AutoConfig, AutoTokenizer, AutoModelForCausalLM
from lightning.pytorch import seed_everything

from videollama2.conversation import conv_templates, SeparatorStyle
from videollama2.model.builder import load_mm_projector

from videollama2.constants import (
    DEFAULT_MMODAL_TOKEN,
    MMODAL_TOKEN_INDEX,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN
)

from videollama2.mm_utils import (
    get_model_name_from_path,
    tokenizer_MMODAL_token,
    KeywordsStoppingCriteria,
    process_video
)

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


# override the code by VideoLLaMA2
def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto",
                          device="cuda", use_flash_attn=False, lora_checkpoint=None, **kwargs):
    if 'token' in kwargs:
        token = kwargs['token']
    else:
        token = None

    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        # kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if "videollama" in model_name.lower() or 'vlb' in model_name.lower():
        # NOTE: lora/qlora model loading
        if 'lora' in model_name.lower() or 'qlora' in model_name.lower():
            if model_base is None:
                cfg_pretrained = PretrainedConfig.from_pretrained(model_path, token=token)
                # NOTE: AutoConfig will modify `_name_or_path` property to `model_path` if `model_path` is not None.
                # cfg_pretrained = AutoConfig.from_pretrained(model_path, token=token)
                model_base = model_base if model_base is not None else cfg_pretrained._name_or_path

            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            # NOTE: remove qlora training quantization config
            if hasattr(lora_cfg_pretrained, 'quantization_config'):
                del lora_cfg_pretrained.quantization_config
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, token=token)
            print('Loading VideoLLaMA from base model...')

            if 'vicuna' in model_base.lower():
                model = Videollama2LlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                                    config=lora_cfg_pretrained, **kwargs)
            elif 'mistral' in model_base.lower():
                model = Videollama2MistralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                                      config=lora_cfg_pretrained, **kwargs)
            else:
                from videollama2.model import Videollama2MistralForCausalLM
                model = Videollama2MistralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                                      config=lora_cfg_pretrained, **kwargs)

            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(
                    torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(
                    torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional VideoLLaMA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'),
                                                 map_location='cpu')
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
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in
                                   non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in
                                       non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, lora_checkpoint if lora_checkpoint is not None else model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None or '-base' in model_name.lower():
            # NOTE: Base/Pretrain model loading
            print('Loading VideoLLaMA 2 from base model...')
            cfg_pretrained = PretrainedConfig.from_pretrained(model_path, token=token)
            # NOTE: AutoConfig will modify `_name_or_path` property to `model_path` if `model_path` is not None.
            # cfg_pretrained = AutoConfig.from_pretrained(model_path, token=token)
            model_base = model_base if model_base is not None else cfg_pretrained._name_or_path

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, token=token)

            if 'vicuna' in model_base.lower():
                model = Videollama2LlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                                    config=cfg_pretrained, **kwargs)
            elif 'mistral' in model_base.lower():
                model = Videollama2MistralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                                      config=cfg_pretrained, **kwargs)
            elif 'mixtral' in model_base.lower():
                model = Videollama2MixtralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                                      config=cfg_pretrained, **kwargs)
            elif 'qwen2' in model_base.lower():
                model = Videollama2Qwen2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                                    config=cfg_pretrained, **kwargs)
            else:
                model = Videollama2MistralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                                      config=cfg_pretrained, **kwargs)

            # NOTE; loading vision-language projector
            # * old codes for loading local mm_projector.bin
            # mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            # mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            # model.load_state_dict(mm_projector_weights, strict=False)
            # * new codes which supports loading mm_projector.bin both offline and online
            mm_projector_weights = load_mm_projector(model_path, token=token)
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            # NOTE: SFT model loading
            cfg_pretrained = PretrainedConfig.from_pretrained(model_path, token=token)
            model_base = cfg_pretrained._name_or_path

            if 'vicuna' in model_base.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, token=token)
                model = Videollama2LlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_base.lower():
                from videollama2.model.language_model.videollama2_mistral import Videollama2MistralForCausalLM
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, token=token)
                model = Videollama2MistralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mixtral' in model_base.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, token=token)
                model = Videollama2MixtralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'qwen2' in model_base.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, token=token)
                model = Videollama2Qwen2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                # NOTE: mistral-based model is our default model.
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, token=token)
                model = Videollama2MistralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
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
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    processor = None

    if "videollama" in model_name.lower() or 'vlb' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        # NOTE: videollama2 adopts the same processor for processing image and video.
        processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, processor, context_len


def initialize_model():
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, None, model_name,
                                                                     load_8bit=args.load_8bit,
                                                                     load_4bit=args.load_4bit,
                                                                     lora_checkpoint=args.lora_checkpoint)
    model.config.num_frames = args.num_video_frames
    return tokenizer, model, processor, context_len


def inference():
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

    tokenizer, model, processor, context_len = initialize_model()
    model.eval()

    # dataloader = accelerator.prepare(dataloader)
    logger.log_info(f"process rank#{args.global_rank}: "
                    f"inference on {len(dataloader)} out of {len(dataset)} dialogues "
                    f"(low resource={args.low_resource})")

    default_mm_token = DEFAULT_MMODAL_TOKEN["VIDEO"]
    modal_token_index = MMODAL_TOKEN_INDEX["VIDEO"]

    results = []
    for i, data in enumerate(dataloader):
        conv = conv_templates[args.conv_mode].copy()
        roles = conv.roles

        video_file = f"{args.resource_dir}/{data['scenario_video']}"
        video = process_video(video_file, processor, model.config.image_aspect_ratio).to(dtype=torch.float16,
                                                                                         device='cuda',
                                                                                         non_blocking=True)
        video = [video]
        modal_list = ['video']

        # logger.log_info_on_main_process(args.global_rank, f"SCENARIO#{i + 1}\n")

        for j, turn in enumerate(data["turns"]):
            inp = turn["question"]
            # logger.log_info_on_main_process(args.global_rank, f"{roles[0]}: {inp}")

            if j == 0:
                inp = default_mm_token + '\n' + inp

            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_MMODAL_token(
                prompt, tokenizer, modal_token_index, return_tensors='pt').unsqueeze(0).cuda()
            attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

            stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE] else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            current_max_len = input_ids.shape[1] + args.max_new_tokens
            if current_max_len - args.max_input_length > 0:
                print('Warning: The number of tokens in current conversation exceeds the max length. '
                      'The model will not see the contexts outside the range.')
            begin_idx = max(0, current_max_len - args.max_input_length)
            input_ids = input_ids[:, begin_idx:]

            # logger.log_info_on_main_process(args.global_rank, f"Prompt:\n---\n{prompt}\n---\n")

            # stopping_criteria
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_masks,
                    images_or_videos=video,
                    modal_list=modal_list,
                    do_sample=args.do_sample,
                    top_p=args.top_p,
                    min_length=args.min_length,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    pad_token_id=tokenizer.eos_token_id,
                )

            pred = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()

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
    parser.add_argument("--model-path", default="DAMO-NLP-SG/VideoLLaMA2-7B")
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--lora-checkpoint", default=None)
    parser.add_argument("--conv-mode", default="llama_2")
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=8)
    parser.add_argument("--res-file", default="", required=True)

    args = parser.parse_args()
    args.global_rank = args.chunk_idx
    args.data_dir = os.path.join(args.resource_dir, "data")
    # args = args_manager.set_args(args)
    args.model_trademark_name = 'videollama2'
    print(f"process rank#{args.global_rank}, {args}")

    seed_everything(args.seed)

    logger = Logger(program=sys.argv[0])

    disable_torch_init()
    inference()

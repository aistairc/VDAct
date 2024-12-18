import sys
import math

import pandas as pd

import evaluate
from datasets import load_dataset
from torch.utils.data import DataLoader
from lightning.pytorch import seed_everything

from video_chatgpt.eval.model_utils import load_video
from video_chatgpt.inference import video_chatgpt_infer
from video_chatgpt.video_conversation import conv_templates, SeparatorStyle
from video_chatgpt.model.utils import KeywordsStoppingCriteria
from transformers import AutoTokenizer, CLIPVisionModel, CLIPImageProcessor, BitsAndBytesConfig, PretrainedConfig, \
    AutoConfig
from video_chatgpt.model import VideoChatGPTLlamaForCausalLM

from logger import Logger
from config import ArgsManager
from utils import *


# Define constants
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def initialize_model(model_base, model_path, lora_checkpoint=None, projection_path=None, load_4bit=False, load_8bit=False, device_map="auto", device=None):
    """
    Initializes the model with given parameters.

    Parameters:
    model_name (str): Name of the model to initialize.
    projection_path (str, optional): Path to the projection weights. Defaults to None.

    Returns:
    tuple: Model, vision tower, tokenizer, image processor, vision config, and video token length.
    """

    kwargs = {"device_map": device_map}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        pass
        # RuntimeError: size mismatch for model.mm_projector.weight
        # kwargs['quantization_config'] = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.float16,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type='nf4'
        # )
    else:
        kwargs['torch_dtype'] = torch.float16

    # Convert model name to user path
    model_base = os.path.expanduser(model_base)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_base)

    # Load model
    model = VideoChatGPTLlamaForCausalLM.from_pretrained(model_base,
                                                         low_cpu_mem_usage=True,
                                                         use_cache=True,
                                                         **kwargs
                                                         )

    # Load image processor
    image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)

    # Set to use start and end tokens for video
    mm_use_vid_start_end = True

    # Add tokens to tokenizer
    tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
    if mm_use_vid_start_end:
        tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)

    # Resize token embeddings of the model
    model.resize_token_embeddings(len(tokenizer))

    # Load the weights from projection_path after resizing the token_embeddings
    if projection_path:
        print(f"Loading weights from {projection_path}")
        status = model.load_state_dict(torch.load(projection_path, map_location='cpu'), strict=False)
        if status.unexpected_keys:
            print(f"Unexpected Keys: {status.unexpected_keys}.\nThe Video-ChatGPT weights are not loaded correctly.")
        print(f"Weights loaded from {projection_path}")

    if "lora" in model_path.lower():
        if model_base is None:
            cfg_pretrained = PretrainedConfig.from_pretrained(model_path)
            # NOTE: AutoConfig will modify `_name_or_path` property to `model_path` if `model_path` is not None.
            # cfg_pretrained = AutoConfig.from_pretrained(model_path, token=token)
            model_base = model_base if model_base is not None else cfg_pretrained._name_or_path

        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
        # NOTE: remove qlora training quantization config
        if hasattr(lora_cfg_pretrained, 'quantization_config'):
            del lora_cfg_pretrained.quantization_config

        print('Loading additional VideoChatGPT weights...')
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

            non_lora_trainables = load_from_hf(lora_checkpoint, 'non_lora_trainables.bin')

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

    model = model.eval()
    # model = model.to(args.global_rank)

    vision_tower_name = "openai/clip-vit-large-patch14"

    # Load vision tower and move to GPU
    vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name, torch_dtype=torch.float16,
                                                   low_cpu_mem_usage=True).to(device)
    vision_tower = vision_tower.eval()

    # Configure vision model
    vision_config = model.get_model().vision_config
    vision_config.vid_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_VIDEO_PATCH_TOKEN])[0]
    vision_config.use_vid_start_end = mm_use_vid_start_end
    if mm_use_vid_start_end:
        vision_config.vid_start_token, vision_config.vid_end_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN])

    # Set video token length
    video_token_len = 356

    return model, vision_tower, tokenizer, image_processor, video_token_len


def get_spatio_temporal_features_torch(features):
    """
    Computes spatio-temporal features from given features.

    Parameters:
    features (torch.Tensor): Input features to process.

    Returns:
    torch.Tensor: Spatio-temporal features.
    """

    # Extract the dimensions of the features
    t, s, c = features.shape

    # Compute temporal tokens as the mean along the time axis
    temporal_tokens = torch.mean(features, dim=1)

    # Padding size calculation
    padding_size = 100 - t

    # Pad temporal tokens if necessary
    if padding_size > 0:
        padding = torch.zeros(padding_size, c, device=features.device)
        temporal_tokens = torch.cat((temporal_tokens, padding), dim=0)

    # Compute spatial tokens as the mean along the spatial axis
    spatial_tokens = torch.mean(features, dim=0)

    # Concatenate temporal and spatial tokens and cast to half precision
    concat_tokens = torch.cat([temporal_tokens, spatial_tokens], dim=0).half()

    return concat_tokens


def inference():
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_base,
                                                                                        args.model_path,
                                                                                        projection_path=args.projection_path,
                                                                                        lora_checkpoint=args.lora_checkpoint,
                                                                                        load_4bit=args.load_4bit,
                                                                                        load_8bit=args.load_8bit,
                                                                                        device=args.global_rank
                                                                                        )
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

    results = []
    for i, data in enumerate(dataloader):
        conv = conv_templates[args.conv_mode].copy()
        roles = conv.roles

        # logger.log_info_on_main_process(args.global_rank, f"SCENARIO#{i + 1}\n")

        # Preprocess video frames and get image tensor
        video_file = f"{args.resource_dir}/{data['scenario_video']}"
        video_frames = load_video(video_file, num_frm=args.num_video_frames)
        image_tensor = image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values']

        # Move image tensor to GPU and reduce precision to half
        image_tensor = image_tensor.half().to(args.global_rank)

        # Generate video spatio-temporal features
        with torch.no_grad():
            image_forward_outs = vision_tower(image_tensor, output_hidden_states=True)
            frame_features = image_forward_outs.hidden_states[-2][:, 1:]  # Use second to last layer as in LLaVA
        video_spatio_temporal_features = get_spatio_temporal_features_torch(frame_features).to(args.global_rank)

        for j, turn in enumerate(data["turns"]):
            inp = turn["question"]
            # logger.log_info_on_main_process(args.global_rank, f"{roles[0]}: {inp}")

            if j == 0:
                if model.get_model().vision_config.use_vid_start_end:
                    inp = (inp + '\n' + DEFAULT_VID_START_TOKEN +
                           DEFAULT_VIDEO_PATCH_TOKEN * video_token_len +
                           DEFAULT_VID_END_TOKEN)
                else:
                    inp = inp + '\n' + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len

            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # Tokenize the prompt
            inputs = tokenizer([prompt])

            # Move inputs to GPU
            input_ids = torch.as_tensor(inputs.input_ids).to(args.global_rank)

            current_max_len = input_ids.shape[1] + args.max_new_tokens
            if current_max_len - args.max_input_length > 0:
                print("Warning: The number of tokens in current conversation exceeds the max length. "
                      f"The model will not see the contexts outside the range. {args.global_rank} {current_max_len} > {args.max_input_length}")

            begin_idx = max(0, current_max_len - args.max_input_length)
            input_ids = input_ids[:, begin_idx:]

            # Define stopping criteria for generation
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

            # logger.log_info_on_main_process(args.global_rank, f"Prompt:\n---\n{prompt}\n---\n")

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    video_spatio_temporal_features=video_spatio_temporal_features.unsqueeze(0),
                    do_sample=True,
                    top_p=args.top_p,
                    min_length=args.min_length,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            # Check if output is the same as input
            n_diff_input_output = (input_ids != output_ids[:, :input_ids.shape[1]]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

            # Decode output tokens
            pred = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

            # Clean output string
            pred = pred.strip().rstrip(stop_str).strip()
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
    parser.add_argument('--model-path', default='mmaaz60/LLaVA-7B-Lightening-v1-1')
    parser.add_argument("--model-base", default='mmaaz60/LLaVA-7B-Lightening-v1-1')
    parser.add_argument("--lora-checkpoint", default=None)
    parser.add_argument('--projection-path', default='Video-ChatGPT/Video-ChatGPT-7B/video_chatgpt-7B.bin')
    parser.add_argument('--cache-dir', default='Video-ChatGPT/cache_dir')
    parser.add_argument("--conv-mode", default="video-chatgpt_v1")
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=8)
    parser.add_argument("--res-file", default="", required=True)

    args = parser.parse_args()
    args.global_rank = args.chunk_idx
    args.data_dir = os.path.join(args.resource_dir, "data")
    # args = args_manager.set_args(args)
    args.model_trademark_name = 'videochatgpt'
    print(f"process rank#{args.global_rank}, {args}")

    seed_everything(args.seed)

    logger = Logger(program=sys.argv[0])

    disable_torch_init()
    inference()

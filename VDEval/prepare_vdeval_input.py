import os
import json
from fire import Fire

from prompt import VDEVAL_INSTRUCTION


def main(
    current_turn,
    model_output_files,
    jsonl_dir,
    summary_dir,
):

    if not os.path.exists(jsonl_dir):
        os.makedirs(jsonl_dir)

    model_output_files = [x.strip() for x in model_output_files.split(",")]
    target_qas = {}
    for model_output_file in model_output_files:
        model_name = model_output_file.strip(".json")
        with open(model_output_file) as f:
            model_outputs = json.load(f)

        for out in model_outputs:
            custom_id = f"{model_name}-{out['id']}"
            if out["turn_num"] == current_turn:
                target_qas[custom_id] = out

    json_lst = []
    if current_turn == 1:
        for k, qa in target_qas.items():
            scenario_id = qa["dial_id"][:-2]
            with open(os.path.join(summary_dir, f"{scenario_id}.txt")) as f:
                summary = f.read()
            user_content = f"Summary: {summary}\nQuestion: {qa['question']}\nReference answer: {qa['ref_answer']}\nCandidate answer: {qa['gen_answer']}\nOutput: "
            json_lst.append({
                "custom_id": k,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini-2024-07-18",
                    "messages":
                    [
                        {"role": "system", "content": VDEVAL_INSTRUCTION},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0,
                    "seed": 42,
                },
            })
    else:
        with open(os.path.join(jsonl_dir, f"input-turn-{current_turn-1}.jsonl")) as f:
            llm_prev_inputs = [json.loads(l) for l in f.readlines()]
        with open(os.path.join(jsonl_dir, f"output-turn-{current_turn-1}.jsonl")) as f:
            llm_prev_outputs = [json.loads(l) for l in f.readlines()]

        for k, qa in target_qas.items():
            prev_id = f"{k[:-2]}{current_turn-1:02}"
            prev_input = [x for x in llm_prev_inputs if x["custom_id"] == prev_id]
            prev_output = [x for x in llm_prev_outputs if x["custom_id"] == prev_id]
            assert len(prev_input) == 1
            assert len(prev_output) == 1
            user_content = prev_input[0]["body"]["messages"][1]["content"]
            out = prev_output[0]["response"]["body"]["choices"][0]["message"]["content"]

            prev_input[0]["custom_id"] = k
            prev_input[0]["body"]["messages"][1]["content"] = f"{user_content}{out}\n\nQuestion: {qa['question']}\nReference answer: {qa['ref_answer']}\nCandidate answer: {qa['gen_answer']}\nOutput: "

            json_lst.append(prev_input[0])

    with open(os.path.join(jsonl_dir, f"input-turn-{current_turn}.jsonl"), "w") as f:
        for l in json_lst:
            json.dump(l, f)
            f.write("\n")

Fire(main)

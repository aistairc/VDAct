import json
from fire import Fire

def main(
    model_output_file,
    eval_output_file,
):
    model_name = model_output_file.strip(".json")
    with open(model_output_file) as f:
        model_outputs = json.load(f)
    with open(eval_output_file) as f:
        eval_outputs = [json.loads(l) for l in f.readlines()]

    custom_id_to_eval_res = {}
    for l in eval_outputs:
        custom_id = l["custom_id"]

        # Convert response to a Python dictionary.
        response_message = l["response"]["body"]["choices"][0]["message"]["content"]
        i_rationale, i_rating = response_message.split("So rating=") if "So rating=" in response_message else (response_message, 1)
        try:
            i_rating = int(i_rating)
        except ValueError:
            i_rating = 1
        custom_id_to_eval_res[custom_id] = {"rationale": i_rationale.strip(), "score": i_rating}

    all_res = {}
    for l in model_outputs:
        custom_id = f"{model_name}-{l['id']}"
        all_res[custom_id] = [
            custom_id_to_eval_res[custom_id],
            {
                "q": l["question"],
                "a": l["ref_answer"],
                "pred": l["gen_answer"],
            }
        ]

    with open(f"vdeval-{model_output_file}", "w") as f:
        json.dump(all_res, f, ensure_ascii=False, indent=4)

    # Calc averaged score
    all_scores = []
    for k, v in all_res.items():
        all_scores.append(v[0]["score"])
    print(f"Num samples: {len(all_scores)}")
    print(f"Averaged score: {sum(all_scores) / len(all_scores)}")
    print(f"Averaged score percentile: {(sum(all_scores) / len(all_scores) - 1) / 2}")

Fire(main)

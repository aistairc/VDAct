import glob
import os
import time
from fire import Fire

from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def main(
    current_turn,
    jsonl_dir,
):
    print(f'-- current turn: {current_turn}')

    batch_input_file = client.files.create(
        file=open(os.path.join(jsonl_dir, f"input-turn-{current_turn}.jsonl"), "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file.id

    # submit batch
    batch_obj = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f'-- batch submitted')

    start = time.time()

    # wait til completed
    while True:
        time.sleep(30)

        running_batch_obj = client.batches.retrieve(batch_obj.id)
        if running_batch_obj.status == "failed":
            print('-- failed, exit')
        if running_batch_obj.status == "completed":
            if running_batch_obj.output_file_id is not None:
                file_response = client.files.content(running_batch_obj.output_file_id)
                output_file = os.path.join(jsonl_dir, f"output-turn-{current_turn}.jsonl")
                with open(output_file, "w") as output_file:
                    output_file.write(file_response.text)
            else:
                raise ValueError
            print(f"-- completed, {(time.time() - start) / 60:.1f} mins")
            break

Fire(main)


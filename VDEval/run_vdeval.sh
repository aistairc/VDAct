#!/bin/bash

jsonl_dir=gpt-4o-8f
model_output_file=test-1.0-gpt-4o-2024-08-06-visual-8.json

if [ ! -d $jsonl_dir ]; then
    mkdir -p $jsonl_dir
fi

for turn in `seq 1 15`;do
    python prepare_vdeval_input.py \
        --current_turn $turn \
        --model_output_files $model_output_file \
        --jsonl_dir $jsonl_dir \
        --summary_dir scenario_refine_summaries

    python submit_batch.py \
        --current_turn $turn \
        --jsonl_dir $jsonl_dir

    cat $jsonl_dir/output-turn-$turn.jsonl >> $jsonl_dir/output-turn-all.jsonl
done

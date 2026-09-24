.venv/bin/python /Users/chongyanghe/Desktop/金融_Agent/format_sft_data.py \
  --input /Users/chongyanghe/Desktop/金融_Agent/finance-agent-main/logs/finance/glm-5.2-v6 \
  --output /Users/chongyanghe/Desktop/金融_Agent/trajectory_v6.jsonl \
  --mode trajectory \
  --output-schema ms-swift \
  --include-reasoning \
  --include-tool-metadata \
  --success-only

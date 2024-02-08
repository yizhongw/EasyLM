set -ex

python conversion_scripts/convert_preference_data.py --input_dataset nvidia/HelpSteer --split train --output converted_pref_data/helpsteer.jsonl

python conversion_scripts/convert_preference_data.py --input_dataset berkeley-nest/Nectar --split train --output converted_pref_data/nectar.jsonl

python conversion_scripts/convert_preference_data.py --input_dataset argilla/ultrafeedback-binarized-preferences-cleaned --split train --output converted_pref_data/ultrafeedback_mean_aspects_cleaned.jsonl

python conversion_scripts/convert_preference_data.py --input_dataset HuggingFaceH4/ultrafeedback_binarized --split train_prefs --output converted_pref_data/ultrafeedback_overall_cleaned.jsonl

python conversion_scripts/convert_preference_data.py --input_dataset stanfordnlp/SHP-2 --split train --output converted_pref_data/shp_2.jsonl

python conversion_scripts/convert_preference_data.py --input_dataset Intel/orca_dpo_pairs --split train --output converted_pref_data/orca_dpo_pairs.jsonl

python conversion_scripts/convert_preference_data.py --input_dataset lvwerra/stack-exchange-paired --split train --output converted_pref_data/stack_exchange_paired.jsonl

python conversion_scripts/convert_preference_data.py --input_dataset Anthropic/hh-rlhf --split train --output converted_pref_data/anthropic_hh.jsonl

# combine and upload to hf, where each dataset is a diff split.
# python conversion_scripts/combine_preference_data.py --input_dir converted_pref_data --output_dir combined_pref_data
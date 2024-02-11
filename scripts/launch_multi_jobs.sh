set -x


# MODEL_NAME="llama_70b_implicit_ft_adaptive_5000"
# python3 -m EasyLM.models.llama.llama_train \
#     --mesh_dim='-1,16,16' \
#     --dtype='bf16' \
#     --num_epochs=2 \
#     --log_freq=1 \
#     --save_model_freq=0 \
#     --save_milestone_freq=0 \
#     --load_llama_config='70b' \
#     --update_llama_config='' \
#     --load_dataset_state='' \
#     --load_checkpoint='params::gs://hamishi-east1/easylm/llama2/70b' \
#     --tokenizer.vocab_file='gs://hamishi-east1/easylm/llama/tokenizer.model' \
#     --optimizer.type='adamw' \
#     --optimizer.adamw_optimizer.weight_decay=0.0 \
#     --optimizer.adamw_optimizer.lr=5e-6 \
#     --optimizer.adamw_optimizer.end_lr=0 \
#     --optimizer.adamw_optimizer.warmup_ratio=0.03 \
#     --optimizer.accumulate_gradient_steps=4 \
#     --train_dataset.type='prompt_completion' \
#     --train_dataset.text_processor.fields='[prompt],completion' \
#     --train_dataset.json_torch_dataset.path='../tsqa/reformatted_train_set/llama2_70B/adaptive/adaptive.jsonl' \
#     --train_dataset.json_torch_dataset.seq_length=128 \
#     --train_dataset.json_torch_dataset.batch_size=32 \
#     --checkpointer.save_optimizer_state=False \
#     --logger.online=True --logger.project='tsqa' --logger.entity='yizhongw' --logger.prefix=${MODEL_NAME} --logger.prefix_to_id=True \
#     --logger.output_dir="gs://yizhong-east1/tsqa_models/${MODEL_NAME}" &&


# MODEL_NAME="llama_70b_implicit_ft_adaptive_all"
# python3 -m EasyLM.models.llama.llama_train \
#     --mesh_dim='-1,16,16' \
#     --dtype='bf16' \
#     --num_epochs=2 \
#     --log_freq=1 \
#     --save_model_freq=0 \
#     --save_milestone_freq=0 \
#     --load_llama_config='70b' \
#     --update_llama_config='' \
#     --load_dataset_state='' \
#     --load_checkpoint='params::gs://hamishi-east1/easylm/llama2/70b' \
#     --tokenizer.vocab_file='gs://hamishi-east1/easylm/llama/tokenizer.model' \
#     --optimizer.type='adamw' \
#     --optimizer.adamw_optimizer.weight_decay=0.0 \
#     --optimizer.adamw_optimizer.lr=5e-6 \
#     --optimizer.adamw_optimizer.end_lr=0 \
#     --optimizer.adamw_optimizer.warmup_ratio=0.03 \
#     --optimizer.accumulate_gradient_steps=4 \
#     --train_dataset.type='prompt_completion' \
#     --train_dataset.text_processor.fields='[prompt],completion' \
#     --train_dataset.json_torch_dataset.path='../tsqa/reformatted_train_set/llama2_70B/adaptive_all/adaptive.jsonl' \
#     --train_dataset.json_torch_dataset.seq_length=128 \
#     --train_dataset.json_torch_dataset.batch_size=32 \
#     --checkpointer.save_optimizer_state=False \
#     --logger.online=True --logger.project='tsqa' --logger.entity='yizhongw' --logger.prefix=${MODEL_NAME} --logger.prefix_to_id=True \
#     --logger.output_dir="gs://yizhong-east1/tsqa_models/${MODEL_NAME}" &&

pgrep llama | xargs kill -9
MODEL_NAME="llama_70b_implicit_ft_2022_correct_5000"
python3 -m EasyLM.models.llama.llama_train \
    --mesh_dim='-1,16,16' \
    --dtype='bf16' \
    --num_epochs=2 \
    --log_freq=1 \
    --save_model_freq=0 \
    --save_milestone_freq=0 \
    --load_llama_config='70b' \
    --update_llama_config='' \
    --load_dataset_state='' \
    --load_checkpoint='params::gs://hamishi-east1/easylm/llama2/70b' \
    --tokenizer.vocab_file='gs://hamishi-east1/easylm/llama/tokenizer.model' \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.0 \
    --optimizer.adamw_optimizer.lr=5e-6 \
    --optimizer.adamw_optimizer.end_lr=0 \
    --optimizer.adamw_optimizer.warmup_ratio=0.03 \
    --optimizer.accumulate_gradient_steps=4 \
    --train_dataset.type='prompt_completion' \
    --train_dataset.text_processor.fields='[prompt],completion' \
    --train_dataset.json_torch_dataset.path='../tsqa/reformatted_train_set/llama2_70B/correct/2022.jsonl' \
    --train_dataset.json_torch_dataset.seq_length=128 \
    --train_dataset.json_torch_dataset.batch_size=32 \
    --checkpointer.save_optimizer_state=False \
    --logger.online=True --logger.project='tsqa' --logger.entity='yizhongw' --logger.prefix=${MODEL_NAME} --logger.prefix_to_id=True \
    --logger.output_dir="gs://yizhong-east1/tsqa_models/${MODEL_NAME}" &&

pgrep llama | xargs kill -9
MODEL_NAME="llama_70b_implicit_ft_2010_correct_5000"
python3 -m EasyLM.models.llama.llama_train \
    --mesh_dim='-1,16,16' \
    --dtype='bf16' \
    --num_epochs=2 \
    --log_freq=1 \
    --save_model_freq=0 \
    --save_milestone_freq=0 \
    --load_llama_config='70b' \
    --update_llama_config='' \
    --load_dataset_state='' \
    --load_checkpoint='params::gs://hamishi-east1/easylm/llama2/70b' \
    --tokenizer.vocab_file='gs://hamishi-east1/easylm/llama/tokenizer.model' \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.0 \
    --optimizer.adamw_optimizer.lr=5e-6 \
    --optimizer.adamw_optimizer.end_lr=0 \
    --optimizer.adamw_optimizer.warmup_ratio=0.03 \
    --optimizer.accumulate_gradient_steps=4 \
    --train_dataset.type='prompt_completion' \
    --train_dataset.text_processor.fields='[prompt],completion' \
    --train_dataset.json_torch_dataset.path='../tsqa/reformatted_train_set/llama2_70B/correct/2010.jsonl' \
    --train_dataset.json_torch_dataset.seq_length=128 \
    --train_dataset.json_torch_dataset.batch_size=32 \
    --checkpointer.save_optimizer_state=False \
    --logger.online=True --logger.project='tsqa' --logger.entity='yizhongw' --logger.prefix=${MODEL_NAME} --logger.prefix_to_id=True \
    --logger.output_dir="gs://yizhong-east1/tsqa_models/${MODEL_NAME}" &&


pgrep llama | xargs kill -9
MODEL_NAME="llama_70b_implicit_ft_2015_correct_5000"
python3 -m EasyLM.models.llama.llama_train \
    --mesh_dim='-1,16,16' \
    --dtype='bf16' \
    --num_epochs=2 \
    --log_freq=1 \
    --save_model_freq=0 \
    --save_milestone_freq=0 \
    --load_llama_config='70b' \
    --update_llama_config='' \
    --load_dataset_state='' \
    --load_checkpoint='params::gs://hamishi-east1/easylm/llama2/70b' \
    --tokenizer.vocab_file='gs://hamishi-east1/easylm/llama/tokenizer.model' \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.0 \
    --optimizer.adamw_optimizer.lr=5e-6 \
    --optimizer.adamw_optimizer.end_lr=0 \
    --optimizer.adamw_optimizer.warmup_ratio=0.03 \
    --optimizer.accumulate_gradient_steps=4 \
    --train_dataset.type='prompt_completion' \
    --train_dataset.text_processor.fields='[prompt],completion' \
    --train_dataset.json_torch_dataset.path='../tsqa/reformatted_train_set/llama2_70B/correct/2015.jsonl' \
    --train_dataset.json_torch_dataset.seq_length=128 \
    --train_dataset.json_torch_dataset.batch_size=32 \
    --checkpointer.save_optimizer_state=False \
    --logger.online=True --logger.project='tsqa' --logger.entity='yizhongw' --logger.prefix=${MODEL_NAME} --logger.prefix_to_id=True \
    --logger.output_dir="gs://yizhong-east1/tsqa_models/${MODEL_NAME}" &&


pgrep llama | xargs kill -9
MODEL_NAME="llama_70b_implicit_ft_2019_correct_5000"
python3 -m EasyLM.models.llama.llama_train \
    --mesh_dim='-1,16,16' \
    --dtype='bf16' \
    --num_epochs=2 \
    --log_freq=1 \
    --save_model_freq=0 \
    --save_milestone_freq=0 \
    --load_llama_config='70b' \
    --update_llama_config='' \
    --load_dataset_state='' \
    --load_checkpoint='params::gs://hamishi-east1/easylm/llama2/70b' \
    --tokenizer.vocab_file='gs://hamishi-east1/easylm/llama/tokenizer.model' \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.0 \
    --optimizer.adamw_optimizer.lr=5e-6 \
    --optimizer.adamw_optimizer.end_lr=0 \
    --optimizer.adamw_optimizer.warmup_ratio=0.03 \
    --optimizer.accumulate_gradient_steps=4 \
    --train_dataset.type='prompt_completion' \
    --train_dataset.text_processor.fields='[prompt],completion' \
    --train_dataset.json_torch_dataset.path='../tsqa/reformatted_train_set/llama2_70B/correct/2019.jsonl' \
    --train_dataset.json_torch_dataset.seq_length=128 \
    --train_dataset.json_torch_dataset.batch_size=32 \
    --checkpointer.save_optimizer_state=False \
    --logger.online=True --logger.project='tsqa' --logger.entity='yizhongw' --logger.prefix=${MODEL_NAME} --logger.prefix_to_id=True \
    --logger.output_dir="gs://yizhong-east1/tsqa_models/${MODEL_NAME}" &&


pgrep llama | xargs kill -9
MODEL_NAME="llama_70b_implicit_ft_2022_all"
python3 -m EasyLM.models.llama.llama_train \
    --mesh_dim='-1,16,16' \
    --dtype='bf16' \
    --num_epochs=2 \
    --log_freq=1 \
    --save_model_freq=0 \
    --save_milestone_freq=0 \
    --load_llama_config='70b' \
    --update_llama_config='' \
    --load_dataset_state='' \
    --load_checkpoint='params::gs://hamishi-east1/easylm/llama2/70b' \
    --tokenizer.vocab_file='gs://hamishi-east1/easylm/llama/tokenizer.model' \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.0 \
    --optimizer.adamw_optimizer.lr=5e-6 \
    --optimizer.adamw_optimizer.end_lr=0 \
    --optimizer.adamw_optimizer.warmup_ratio=0.03 \
    --optimizer.accumulate_gradient_steps=4 \
    --train_dataset.type='prompt_completion' \
    --train_dataset.text_processor.fields='[prompt],completion' \
    --train_dataset.json_torch_dataset.path='../tsqa/reformatted_train_set/llama2_70B/all/2022.jsonl' \
    --train_dataset.json_torch_dataset.seq_length=128 \
    --train_dataset.json_torch_dataset.batch_size=32 \
    --checkpointer.save_optimizer_state=False \
    --logger.online=True --logger.project='tsqa' --logger.entity='yizhongw' --logger.prefix=${MODEL_NAME} --logger.prefix_to_id=True \
    --logger.output_dir="gs://yizhong-east1/tsqa_models/${MODEL_NAME}" &&


pgrep llama | xargs kill -9
MODEL_NAME="llama_70b_implicit_ft_2023_all"
python3 -m EasyLM.models.llama.llama_train \
    --mesh_dim='-1,16,16' \
    --dtype='bf16' \
    --num_epochs=2 \
    --log_freq=1 \
    --save_model_freq=0 \
    --save_milestone_freq=0 \
    --load_llama_config='70b' \
    --update_llama_config='' \
    --load_dataset_state='' \
    --load_checkpoint='params::gs://hamishi-east1/easylm/llama2/70b' \
    --tokenizer.vocab_file='gs://hamishi-east1/easylm/llama/tokenizer.model' \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.0 \
    --optimizer.adamw_optimizer.lr=5e-6 \
    --optimizer.adamw_optimizer.end_lr=0 \
    --optimizer.adamw_optimizer.warmup_ratio=0.03 \
    --optimizer.accumulate_gradient_steps=4 \
    --train_dataset.type='prompt_completion' \
    --train_dataset.text_processor.fields='[prompt],completion' \
    --train_dataset.json_torch_dataset.path='../tsqa/reformatted_train_set/llama2_70B/all/2023.jsonl' \
    --train_dataset.json_torch_dataset.seq_length=128 \
    --train_dataset.json_torch_dataset.batch_size=32 \
    --checkpointer.save_optimizer_state=False \
    --logger.online=True --logger.project='tsqa' --logger.entity='yizhongw' --logger.prefix=${MODEL_NAME} --logger.prefix_to_id=True \
    --logger.output_dir="gs://yizhong-east1/tsqa_models/${MODEL_NAME}" &&

    
# MODEL_NAME="llama_70b_implicit_ft_2023_correct_5000"
# python3 -m EasyLM.models.llama.llama_train \
#     --mesh_dim='-1,16,16' \
#     --dtype='bf16' \
#     --num_epochs=2 \
#     --log_freq=1 \
#     --save_model_freq=0 \
#     --save_milestone_freq=0 \
#     --load_llama_config='70b' \
#     --update_llama_config='' \
#     --load_dataset_state='' \
#     --load_checkpoint='params::gs://hamishi-east1/easylm/llama2/70b' \
#     --tokenizer.vocab_file='gs://hamishi-east1/easylm/llama/tokenizer.model' \
#     --optimizer.type='adamw' \
#     --optimizer.adamw_optimizer.weight_decay=0.0 \
#     --optimizer.adamw_optimizer.lr=5e-6 \
#     --optimizer.adamw_optimizer.end_lr=0 \
#     --optimizer.adamw_optimizer.warmup_ratio=0.03 \
#     --optimizer.accumulate_gradient_steps=4 \
#     --train_dataset.type='prompt_completion' \
#     --train_dataset.text_processor.fields='[prompt],completion' \
#     --train_dataset.json_torch_dataset.path='../tsqa/reformatted_train_set/llama2_70B/correct/2023.jsonl' \
#     --train_dataset.json_torch_dataset.seq_length=128 \
#     --train_dataset.json_torch_dataset.batch_size=32 \
#     --checkpointer.save_optimizer_state=False \
#     --logger.online=True --logger.project='tsqa' --logger.entity='yizhongw' --logger.prefix=${MODEL_NAME} --logger.prefix_to_id=True \
#     --logger.output_dir="gs://yizhong-east1/tsqa_models/${MODEL_NAME}" &&


pgrep llama | xargs kill -9
MODEL_NAME="llama_70b_implicit_ft_2023_random_5000"
python3 -m EasyLM.models.llama.llama_train \
    --mesh_dim='-1,16,16' \
    --dtype='bf16' \
    --num_epochs=2 \
    --log_freq=1 \
    --save_model_freq=0 \
    --save_milestone_freq=0 \
    --load_llama_config='70b' \
    --update_llama_config='' \
    --load_dataset_state='' \
    --load_checkpoint='params::gs://hamishi-east1/easylm/llama2/70b' \
    --tokenizer.vocab_file='gs://hamishi-east1/easylm/llama/tokenizer.model' \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.0 \
    --optimizer.adamw_optimizer.lr=5e-6 \
    --optimizer.adamw_optimizer.end_lr=0 \
    --optimizer.adamw_optimizer.warmup_ratio=0.03 \
    --optimizer.accumulate_gradient_steps=4 \
    --train_dataset.type='prompt_completion' \
    --train_dataset.text_processor.fields='[prompt],completion' \
    --train_dataset.json_torch_dataset.path='../tsqa/reformatted_train_set/llama2_70B/random/2023.jsonl' \
    --train_dataset.json_torch_dataset.seq_length=128 \
    --train_dataset.json_torch_dataset.batch_size=32 \
    --checkpointer.save_optimizer_state=False \
    --logger.online=True --logger.project='tsqa' --logger.entity='yizhongw' --logger.prefix=${MODEL_NAME} --logger.prefix_to_id=True \
    --logger.output_dir="gs://yizhong-east1/tsqa_models/${MODEL_NAME}" &&


pgrep llama | xargs kill -9
MODEL_NAME="llama_70b_implicit_ft_2023_popular_5000"
python3 -m EasyLM.models.llama.llama_train \
    --mesh_dim='-1,16,16' \
    --dtype='bf16' \
    --num_epochs=2 \
    --log_freq=1 \
    --save_model_freq=0 \
    --save_milestone_freq=0 \
    --load_llama_config='70b' \
    --update_llama_config='' \
    --load_dataset_state='' \
    --load_checkpoint='params::gs://hamishi-east1/easylm/llama2/70b' \
    --tokenizer.vocab_file='gs://hamishi-east1/easylm/llama/tokenizer.model' \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.0 \
    --optimizer.adamw_optimizer.lr=5e-6 \
    --optimizer.adamw_optimizer.end_lr=0 \
    --optimizer.adamw_optimizer.warmup_ratio=0.03 \
    --optimizer.accumulate_gradient_steps=4 \
    --train_dataset.type='prompt_completion' \
    --train_dataset.text_processor.fields='[prompt],completion' \
    --train_dataset.json_torch_dataset.path='../tsqa/reformatted_train_set/llama2_70B/popular/2023.jsonl' \
    --train_dataset.json_torch_dataset.seq_length=128 \
    --train_dataset.json_torch_dataset.batch_size=32 \
    --checkpointer.save_optimizer_state=False \
    --logger.online=True --logger.project='tsqa' --logger.entity='yizhongw' --logger.prefix=${MODEL_NAME} --logger.prefix_to_id=True \
    --logger.output_dir="gs://yizhong-east1/tsqa_models/${MODEL_NAME}" &&


pgrep llama | xargs kill -9
MODEL_NAME="llama_70b_implicit_ft_2023_confident_5000"
python3 -m EasyLM.models.llama.llama_train \
    --mesh_dim='-1,16,16' \
    --dtype='bf16' \
    --num_epochs=2 \
    --log_freq=1 \
    --save_model_freq=0 \
    --save_milestone_freq=0 \
    --load_llama_config='70b' \
    --update_llama_config='' \
    --load_dataset_state='' \
    --load_checkpoint='params::gs://hamishi-east1/easylm/llama2/70b' \
    --tokenizer.vocab_file='gs://hamishi-east1/easylm/llama/tokenizer.model' \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.0 \
    --optimizer.adamw_optimizer.lr=5e-6 \
    --optimizer.adamw_optimizer.end_lr=0 \
    --optimizer.adamw_optimizer.warmup_ratio=0.03 \
    --optimizer.accumulate_gradient_steps=4 \
    --train_dataset.type='prompt_completion' \
    --train_dataset.text_processor.fields='[prompt],completion' \
    --train_dataset.json_torch_dataset.path='../tsqa/reformatted_train_set/llama2_70B/confident/2023.jsonl' \
    --train_dataset.json_torch_dataset.seq_length=128 \
    --train_dataset.json_torch_dataset.batch_size=32 \
    --checkpointer.save_optimizer_state=False \
    --logger.online=True --logger.project='tsqa' --logger.entity='yizhongw' --logger.prefix=${MODEL_NAME} --logger.prefix_to_id=True \
    --logger.output_dir="gs://yizhong-east1/tsqa_models/${MODEL_NAME}" && 
import jax
jax.devices()  # sync the tpus...

import pprint
import math

from tqdm import tqdm, trange
import mlxu

import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit, with_sharding_constraint
from jax.sharding import PartitionSpec as PS
import flax
from flax import linen as nn
from flax.jax_utils import prefetch_to_device
from flax.training.train_state import TrainState
import torch
from jax.sharding import NamedSharding
from jax.experimental import multihost_utils

from EasyLM.data import DatasetFactory
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.optimizers import OptimizerFactory
from EasyLM.jax_utils import (
    JaxRNG, get_jax_mp_mesh, next_rng, match_partition_rules,
    cross_entropy_loss_and_accuracy, global_norm,
    set_random_seed, get_weight_decay_mask,
    make_shard_and_gather_fns, global_mean, global_max,
    difference, average_metrics
)
from EasyLM.models.llama.llama_model import (
    LLaMAConfig, FlaxLLaMAForCausalLMModule
)




FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    initialize_jax_distributed=False,
    mp_mesh_dim='-1,1',
    num_epochs=0,
    total_steps=10000,
    load_llama_config='',
    update_llama_config='',
    load_checkpoint='',
    load_dataset_state='',
    log_freq=50,
    save_model_freq=0,
    save_milestone_freq=0,
    eval_steps=0,
    tokenizer=LLaMAConfig.get_tokenizer_config(),
    train_dataset=DatasetFactory.get_default_config(),
    eval_dataset=DatasetFactory.get_default_config(),
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    llama=LLaMAConfig.get_default_config(),
    logger=mlxu.WandBLogger.get_default_config(),
    log_all_worker=False,
)


def main(argv):
    if FLAGS.initialize_jax_distributed:
        jax.distributed.initialize()

    mesh = get_jax_mp_mesh(FLAGS.mp_mesh_dim)
    # # work out current data-parallel shard
    # dummy_sample = jnp.zeros((8, 512), dtype=jnp.int32)
    # from jax.sharding import PositionalSharding
    # batch = jax.device_put(dummy_sample, PositionalSharding(mesh, PS('dp')))
    # print(jax.debug.visualize_array_sharding(batch))

    variant = mlxu.get_user_flags(FLAGS, FLAGS_DEF)
    flags_config_dict = mlxu.user_flags_to_config_dict(FLAGS, FLAGS_DEF)
    logger = mlxu.WandBLogger(
        config=FLAGS.logger,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax.process_index() == 0),
    )
    set_random_seed(FLAGS.seed)

    print("Loading dataset...")
    if FLAGS.load_dataset_state != '':
        dataset = mlxu.load_pickle(FLAGS.load_dataset_state)
    else:
        tokenizer = LLaMAConfig.get_tokenizer(FLAGS.tokenizer)
        dataset = DatasetFactory.load_dataset(FLAGS.train_dataset, tokenizer)

    if isinstance(dataset, torch.utils.data.DataLoader):
        wrapped_dataset = dataset.dataset
    else:
        wrapped_dataset = dataset

    real_batch_size = wrapped_dataset.config.batch_size
    steps_per_epoch = len(wrapped_dataset) // real_batch_size

    if FLAGS.eval_steps > 0:
        eval_dataset = DatasetFactory.load_dataset(
            FLAGS.eval_dataset, tokenizer
        )

    seq_length = wrapped_dataset.seq_length

    print("Building model...")
    if FLAGS.load_llama_config != '':
        llama_config = LLaMAConfig.load_config(FLAGS.load_llama_config)
    else:
        llama_config = LLaMAConfig(**FLAGS.llama)

    if FLAGS.update_llama_config != '':
        llama_config.update(dict(eval(FLAGS.update_llama_config)))

    llama_config.update(dict(
        bos_token_id=wrapped_dataset.tokenizer.bos_token_id,
        eos_token_id=wrapped_dataset.tokenizer.eos_token_id,
    ))
    if llama_config.vocab_size < wrapped_dataset.vocab_size:
        llama_config.update(dict(vocab_size=wrapped_dataset.vocab_size))
    model = FlaxLLaMAForCausalLMModule(llama_config)

    print("Building optimizer...")
    if FLAGS.num_epochs > 0:
        FLAGS.optimizer.adamw_optimizer.lr_decay_steps = FLAGS.num_epochs * steps_per_epoch

    optimizer, optimizer_info = OptimizerFactory.get_optimizer(
        FLAGS.optimizer,
        get_weight_decay_mask(LLaMAConfig.get_weight_decay_exclusions())
    )

    def create_trainstate_from_params(params):
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(llama_config.rng_keys()),
        )
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def eval_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        tokens = with_sharding_constraint(batch['tokens'], PS('dp'))
        loss_masks = with_sharding_constraint(batch['loss_masks'], PS('dp'))
        bos_tokens = jnp.full(
            (tokens.shape[0], 1), llama_config.bos_token_id, dtype=jnp.int32
        )
        inputs = jnp.concatenate([bos_tokens, tokens[:, :-1]], axis=1)
        logits = model.apply(
            train_state.params, inputs, deterministic=True,
            rngs=rng_generator(llama_config.rng_keys()),
        ).logits
        loss, accuracy = cross_entropy_loss_and_accuracy(logits, tokens, loss_masks)
        metrics = dict(
            eval_loss=loss,
            eval_accuracy=accuracy,
        )
        return rng_generator(), metrics

    def train_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        jax.debug.inspect_array_sharding(batch['tokens'], callback=print)
        tokens = with_sharding_constraint(batch['tokens'], PS('dp'))
        attention_masks = with_sharding_constraint(batch['attention_masks'], PS('dp'))
        loss_masks = with_sharding_constraint(batch['loss_masks'], PS('dp'))
        jax.debug.inspect_array_sharding(tokens, callback=print)

        def loss_and_accuracy(params):
            bos_tokens = jnp.full(
                (tokens.shape[0], 1), llama_config.bos_token_id, dtype=jnp.int32
            )
            attention_mask = jnp.concatenate([jnp.ones_like(bos_tokens), attention_masks], axis=1, dtype=jnp.int32)
            inputs = jnp.concatenate([bos_tokens, tokens[:, :-1]], axis=1)
            logits = model.apply(
                params, inputs, attention_mask=attention_mask[:, :-1], deterministic=False,
                rngs=rng_generator(llama_config.rng_keys()),
            ).logits
            return cross_entropy_loss_and_accuracy(logits, tokens, loss_masks)
        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (loss, (accuracy, valid_text_length)), grads = grad_fn(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        metrics = dict(
            loss=loss,
            accuracy=accuracy,
            learning_rate=optimizer_info['learning_rate_schedule'](train_state.step),
            # gradient_norm=global_norm(grads),
            # gradient_mean=global_mean(grads),
            # param_mean=global_mean(train_state.params),
            # param_norm=global_norm(train_state.params),
            # param_max=global_max(train_state.params),
            # update_max=global_max(update),
            # update_mean=global_mean(update),
            # valid_text_length=valid_text_length,
        )
        return train_state, rng_generator(), metrics

    # grab one batch to get shape
    token_inp = (wrapped_dataset.config.batch_size, wrapped_dataset.config.seq_length)
    batch_shape = {
        'tokens': jax.ShapeDtypeStruct(shape=token_inp, dtype='i4'),
        'loss_masks': jax.ShapeDtypeStruct(shape=token_inp, dtype='float32'),
        'attention_masks': jax.ShapeDtypeStruct(shape=token_inp, dtype='i4'),
    }
    batch_spec = {
        'tokens': PS("dp", None),
        'loss_masks': PS("dp", None),
        'attention_masks': PS("dp", None),
    }
    # from EasyLM.data import create_device_to_index, partition_data_on_hosts
    # device_index = create_device_to_index(mesh, batch_shape, batch_spec)

    print("Initializing training state and pjitting...")
    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    train_state_partition = match_partition_rules(
        LLaMAConfig.get_partition_rules(), train_state_shapes
    )

    shard_fns, gather_fns = make_shard_and_gather_fns(
        train_state_partition, train_state_shapes
    )
    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer, logger.output_dir,
        enable=jax.process_index() == 0,
    )

    sharded_init_fn = pjit(
        init_fn,
        in_shardings=PS(),
        out_shardings=train_state_partition
    )

    sharded_create_trainstate_from_params = pjit(
        create_trainstate_from_params,
        in_shardings=(train_state_partition.params, ),
        out_shardings=train_state_partition,
        donate_argnums=(0, ),
    )

    sharded_train_step = pjit(
        train_step,
        in_shardings=(train_state_partition, None, batch_spec),
        out_shardings=(train_state_partition, None, None),
        donate_argnums=(0, 1),
    )

    sharded_eval_step = pjit(
        eval_step,
        in_shardings=(train_state_partition, PS(), PS()),
        out_shardings=(PS(), PS()),
        donate_argnums=(1,),
    )

    def save_checkpoint(train_state, milestone=False):
        step = int(jax.device_get(train_state.step))
        metadata = dict(
            step=step,
            variant=variant,
            flags=flags_config_dict,
            llama_config=llama_config.to_dict(),
        )
        checkpointer.save_all(
            train_state=train_state,
            gather_fns=gather_fns,
            metadata=metadata,
            dataset=dataset,
            milestone=milestone,
        )


    assert len(mesh.shape) == 3, 'MP mesh must be 2D'
    with mesh:
        train_state, restored_params = None, None
        if FLAGS.load_checkpoint != '':
            print("Loading checkpoint... (may take time to download)")
            train_state, restored_params = checkpointer.load_trainstate_checkpoint(
                FLAGS.load_checkpoint, train_state_shapes, shard_fns
            )
            print("Checkpoint loaded.")

        if train_state is None and restored_params is None:
            # Initialize from scratch
            train_state = sharded_init_fn(next_rng())
        elif train_state is None and restored_params is not None:
            # Restore from params but initialize train_state
            train_state = sharded_create_trainstate_from_params(restored_params)
            del restored_params

        start_step = int(jax.device_get(train_state.step))

        # if FLAGS.save_model_freq > 0:
        #     print("Initial save...")
        #     save_checkpoint(train_state)

        sharded_rng = next_rng()

        if FLAGS.num_epochs > 0:
            epoch_counter = trange(0, FLAGS.num_epochs, ncols=0, position=0)
            step_counter = trange(start_step, steps_per_epoch, ncols=0, position=1)
        else:
            epoch_counter = trange(0, math.ceil(FLAGS.total_steps / steps_per_epoch), ncols=0, position=0)
            step_counter = trange(start_step, FLAGS.total_steps, ncols=0, position=1)

        for epoch in epoch_counter:
            for step, batch in zip(step_counter, dataset):
                if isinstance(batch, (list, tuple)):
                    batch = {
                        'tokens': batch[0],
                        'loss_masks': batch[1],
                        'attention_masks': batch[2],
                    }
                # I can probably do some smart pjitting for this...
                # batch = partition_data_on_hosts(
                #     batch, device_index, batch_shape, mesh, batch_spec
                # )
                # for k in batch:
                #     batch[k] = jax.device_put(batch[k], NamedSharding(mesh, PS('dp')))
               
                def make_array(batch_item, spec):
                    def cb(index):
                        return batch_item[index]
                    return jax.make_array_from_callback(token_inp, jax.sharding.NamedSharding(mesh, spec), cb)
                batch = jax.tree_util.tree_map(make_array, batch, batch_spec)
                jax.debug.visualize_array_sharding(batch['tokens'])
                train_state, sharded_rng, metrics = sharded_train_step(
                    train_state, sharded_rng, batch
                )

                if step % FLAGS.log_freq == 0:
                    if FLAGS.eval_steps > 0:
                        eval_metric_list = []
                        for batch in eval_dataset:
                            if isinstance(batch, (list, tuple)):
                                batch = {
                                    'tokens': batch[0],
                                    'loss_masks': batch[1],
                                    'attention_masks': batch[2],
                                }
                            sharded_rng, eval_metrics = sharded_eval_step(
                                train_state, sharded_rng, batch
                            )
                            eval_metric_list.append(eval_metrics)
                        metrics.update(average_metrics(eval_metric_list))
                    log_metrics = {"step": step}
                    log_metrics = jax.device_get(log_metrics)
                    log_metrics.update(metrics)
                    log_metrics = {k: float(v) for k, v in log_metrics.items()}
                    logger.log(log_metrics)
                    tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

                if FLAGS.save_milestone_freq > 0 and (step + 1) % FLAGS.save_milestone_freq == 0:
                    save_checkpoint(train_state, milestone=True)
                elif FLAGS.save_model_freq > 0 and (step + 1) % FLAGS.save_model_freq == 0:
                    save_checkpoint(train_state)
            # save model at the end of each epoch
            if FLAGS.save_model_freq > 0:
                save_checkpoint(train_state, milestone=True)
            # reset step counter
            if FLAGS.num_epochs > 0:
                step_counter = trange(start_step, steps_per_epoch, ncols=0, position=1)
            else:
                step_counter = trange(start_step, FLAGS.total_steps, ncols=0, position=1)

        # final log
        if FLAGS.log_freq > 0:
            log_metrics = {"step": step}
            metrics = {k: float(v) for k, v in metrics.items()}
            log_metrics.update(metrics)
            logger.log(log_metrics)
            tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")
        if True:#FLAGS.save_model_freq > 0:
            save_checkpoint(train_state, milestone=True)


if __name__ == "__main__":
    mlxu.run(main)

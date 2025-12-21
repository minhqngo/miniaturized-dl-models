import jax
import optax
import time
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from vision_models.tiny_alexnet import TinyAlexNet
from vision_models.tiny_googlenet import TinyGoogLeNet

N_CLASSES = 10
LEARNING_RATE = 3e-4
EPOCHS = 30
BATCH_SIZE = 128
SEED = 42


def normalize_and_transpose(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.transpose(image, [2, 0, 1]) 
    return image, label


def get_dataloaders(batch_size):
    (ds_train, ds_test), ds_info = tfds.load(
        'cifar10',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    ds_train = ds_train.map(normalize_and_transpose, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(normalize_and_transpose, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return tfds.as_numpy(ds_train), tfds.as_numpy(ds_test)

optimizer = optax.adamw(LEARNING_RATE)


def loss_fn(model, state, x, y, key):
    batch_size = x.shape[0]
    keys = jax.random.split(key, batch_size)
    state_batched = jax.tree_util.tree_map(
        lambda s: jnp.broadcast_to(s, (batch_size,) + s.shape), 
        state
    )
    logits, new_state_batched = jax.vmap(
        model, 
        axis_name="batch", 
        in_axes=(0, 0, 0, None) 
    )(x, state_batched, keys, False)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    new_state = jax.tree_util.tree_map(lambda s: s[0], new_state_batched)
    return loss, new_state


@eqx.filter_jit
def train_step(model, state, opt_state, x, y, key):
    (loss, new_state), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, state, x, y, key
    )
    updates, new_opt_state = optimizer.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_state, new_opt_state, loss


@eqx.filter_jit
def eval_step(model, state, x, y):
    logits, _ = jax.vmap(model, axis_name="batch", in_axes=(0, None, None, None))(x, state, None, True)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)


if __name__ == '__main__':
    key = jax.random.PRNGKey(SEED)
    model_key, train_key = jax.random.split(key)
    model, state = eqx.nn.make_with_state(TinyGoogLeNet)(n_classes=N_CLASSES, key=key)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    train_ds, test_ds = get_dataloaders(BATCH_SIZE)
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        epoch_loss = 0.0
        batches = 0

        for batch_x, batch_y in train_ds:
            step_key, train_key = jax.random.split(train_key)
            model, state, opt_state, loss = train_step(
                model, state, opt_state, batch_x, batch_y, step_key
            )
            epoch_loss += loss.item()
            batches += 1

        avg_train_loss = epoch_loss / batches

        total_acc = 0.0
        eval_batches = 0
        for batch_x, batch_y in test_ds:
            acc = eval_step(model, state, batch_x, batch_y)
            total_acc += acc.item()
            eval_batches += 1
        
        val_acc = total_acc / eval_batches
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {avg_train_loss:.4f} | "
              f"Val Acc: {val_acc*100:.2f}% | "
              f"Time: {time.time() - start_time:.2f}s")
        
    eqx.tree_serialise_leaves("tiny_googlenet_cifar10.eqx", model)

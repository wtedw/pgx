import jax
import jax.numpy as jnp

NUM_CHESS_ACTIONS = 4672
NUM_CHESS_WORDS = (NUM_CHESS_ACTIONS + 31) // 32  # 146


def pack_mask(mask: jnp.ndarray) -> jnp.ndarray:
    """Packs a boolean mask of arbitrary length into a compact uint32 bitset."""
    n = mask.shape[0]
    num_words = (n + 31) // 32
    padded = jnp.pad(mask.astype(jnp.uint32), (0, num_words * 32 - n))
    reshaped = padded.reshape(num_words, 32)
    powers_of_2 = jnp.left_shift(jnp.uint32(1), jnp.arange(32, dtype=jnp.uint32))
    return jnp.sum(reshaped * powers_of_2, axis=1, dtype=jnp.uint32)


def unpack_bitmask(bitset: jnp.ndarray) -> jnp.ndarray:
    """Unpacks a uint32 bitset into a boolean mask of length num_words * 32."""
    powers_of_2 = jnp.left_shift(jnp.uint32(1), jnp.arange(32, dtype=jnp.uint32))
    return ((bitset[:, None] & powers_of_2[None, :]) > 0).flatten()


pack_mask_vmap = jax.vmap(pack_mask)
unpack_bitmask_vmap = jax.vmap(unpack_bitmask)

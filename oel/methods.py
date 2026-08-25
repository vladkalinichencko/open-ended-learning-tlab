"""Teacher-side модели вероятности успеха."""

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import torch.nn.functional as functional
from flax import linen as nn
from flax.training.train_state import TrainState
from torchvision.models import ResNet18_Weights, resnet18

import levels as maze_features

class FeaturePredictor(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(32)(x))
        return nn.Dense(1)(x).squeeze(-1)

class GridPredictor(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(16, (3, 3), padding="SAME")(x))
        x = nn.relu(nn.Conv(32, (3, 3), padding="SAME")(x))
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(nn.Dense(32)(x))
        return nn.Dense(1)(x).squeeze(-1)

class TransitionPredictor(nn.Module):
    @nn.compact
    def __call__(self, observations, actions):
        time, batch = observations.shape[:2]
        x = observations.reshape((-1, 5, 5, 3))
        x = nn.relu(nn.Conv(32, (3, 3), padding="SAME")(x))
        x = nn.relu(nn.Conv(64, (3, 3), padding="SAME")(x))
        x = nn.Dense(128)(x.reshape((time, batch, -1)))
        x = jnp.concatenate((x, actions[..., None].astype(jnp.float32)), axis=-1)
        x = nn.RNN(nn.OptimizedLSTMCell(128), time_major=True)(x)
        x = nn.Dense(64 * 5 * 5)(x).reshape((-1, 5, 5, 64))
        x = nn.relu(nn.ConvTranspose(32, (3, 3), padding="SAME")(x))
        x = nn.sigmoid(nn.ConvTranspose(3, (3, 3), padding="SAME")(x))
        return x.reshape((time, batch, 5, 5, 3))

def fixed_features(level_batch) -> jnp.ndarray:
    rows = []
    wall_maps = np.asarray(level_batch.wall_map)
    starts = np.asarray(level_batch.agent_pos)
    goals = np.asarray(level_batch.goal_pos)
    for walls, start, goal in zip(wall_maps, starts, goals):
        row = maze_features.features(
            walls,
            (int(start[1]), int(start[0])),
            (int(goal[1]), int(goal[0])),
        )
        rows.append([
            row["путь"] / 168,
            row["прямая"] / 24,
            row["извилистость"] / 168,
            row["тупиков"] / 169,
            row["развилок"] / 169,
            row["плотность стен"],
            row["длина коридора"] / 169,
        ])
    return jnp.asarray(rows, dtype=jnp.float32)

def grid_features(level_batch) -> jnp.ndarray:
    walls = jnp.asarray(level_batch.wall_map, dtype=jnp.float32)
    starts = jax.nn.one_hot(level_batch.agent_pos[:, 1] * 13 + level_batch.agent_pos[:, 0], 169)
    goals = jax.nn.one_hot(level_batch.goal_pos[:, 1] * 13 + level_batch.goal_pos[:, 0], 169)
    return jnp.stack((walls, starts.reshape(-1, 13, 13), goals.reshape(-1, 13, 13)), axis=-1)

def create_predictor(kind: str, seed: int, learning_rate: float) -> tuple[TrainState, Callable]:
    if kind == "fixed":
        model, example, encode = FeaturePredictor(), jnp.zeros((1, 7)), fixed_features
    elif kind == "cnn":
        model, example, encode = GridPredictor(), jnp.zeros((1, 13, 13, 3)), grid_features
    else:
        raise ValueError(f"Unknown predictor: {kind}")
    params = model.init(jax.random.PRNGKey(seed), example)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(learning_rate))
    return state, encode

@jax.jit
def update_predictor(state: TrainState, x: jnp.ndarray, success: jnp.ndarray):
    def loss(params):
        logits = state.apply_fn(params, x)
        return optax.sigmoid_binary_cross_entropy(logits, success).mean()

    value, gradients = jax.value_and_grad(loss)(state.params)
    return state.apply_gradients(grads=gradients), value

@jax.jit
def predict_success(state: TrainState, x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.sigmoid(state.apply_fn(state.params, x))

def create_transition_predictor(seed: int) -> TrainState:
    model = TransitionPredictor()
    params = model.init(
        jax.random.PRNGKey(seed),
        jnp.zeros((1, 1, 5, 5, 3)),
        jnp.zeros((1, 1), dtype=jnp.int32),
    )
    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adamw(1e-4, weight_decay=1e-5))
    return TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

@jax.jit
def update_transition_predictor(state: TrainState, observations, actions):
    inputs, targets, actions = observations.image[:-1], observations.image[1:], actions[:-1]

    def loss(params):
        predictions = state.apply_fn(params, inputs, actions)
        per_level = jnp.abs(predictions - targets).mean(axis=(0, 2, 3, 4))
        return per_level.mean(), (per_level, predictions)

    (value, (per_level, predictions)), gradients = jax.value_and_grad(loss, has_aux=True)(state.params)
    return state.apply_gradients(grads=gradients), value, per_level, predictions, targets

@jax.jit
def predict_transitions(state: TrainState, observations, actions):
    return state.apply_fn(state.params, observations.image[:-1], actions[:-1])

class ResNetPredictor:
    """Замороженный ImageNet ResNet-18 и одна обучаемая голова."""

    def __init__(self, seed: int, learning_rate: float):
        self.torch = torch
        torch.manual_seed(seed)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = torch.nn.Identity()
        self.encoder = model.eval().to(self.device)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.head = torch.nn.Linear(512, 1).to(self.device)
        self.optimizer = torch.optim.Adam(self.head.parameters(), lr=learning_rate)
        self.mean = torch.tensor((0.485, 0.456, 0.406), device=self.device)[None, :, None, None]
        self.std = torch.tensor((0.229, 0.224, 0.225), device=self.device)[None, :, None, None]

    def predict_and_update(self, level_batch, success) -> tuple[jnp.ndarray, float]:
        images = np.asarray(grid_features(level_batch)).transpose(0, 3, 1, 2).copy()
        images = self.torch.as_tensor(images, dtype=self.torch.float32, device=self.device)
        images = functional.interpolate(images, size=(224, 224), mode="nearest")
        images = (images - self.mean) / self.std
        with self.torch.no_grad():
            embedding = self.encoder(images)
            probability = self.head(embedding).sigmoid().squeeze(-1)
        labels = self.torch.as_tensor(np.asarray(success).copy(), dtype=self.torch.float32, device=self.device)
        logits = self.head(embedding).squeeze(-1)
        loss = functional.binary_cross_entropy_with_logits(logits, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return jnp.asarray(probability.cpu().numpy()), float(loss.detach().cpu())

"""Tests for RL algorithms."""

import gymnasium as gym
import pytest
import torch

from rl_scaling_laws.algorithms import SAC, DQN, PPO


class TestSAC:
    """Tests for SAC algorithm."""

    def test_initialization(self, continuous_env):
        """Test SAC initialization."""
        sac = SAC(
            observation_space=continuous_env.observation_space,
            action_space=continuous_env.action_space,
            target_params=10_000,
            buffer_size=1000,
        )

        assert sac is not None
        assert sac.actor is not None
        assert sac.critic1 is not None
        assert sac.critic2 is not None

    def test_predict(self, continuous_env):
        """Test action prediction."""
        sac = SAC(
            observation_space=continuous_env.observation_space,
            action_space=continuous_env.action_space,
            target_params=10_000,
        )

        obs, _ = continuous_env.reset()
        action = sac.predict(obs, deterministic=True)

        assert action.shape == continuous_env.action_space.shape

    def test_collect_rollouts(self, continuous_env):
        """Test rollout collection."""
        sac = SAC(
            observation_space=continuous_env.observation_space,
            action_space=continuous_env.action_space,
            target_params=10_000,
            learning_starts=10,
        )

        collected = sac.collect_rollouts(continuous_env, n_steps=100)

        assert collected == 100
        assert len(sac.buffer) == 100

    def test_train(self, continuous_env):
        """Test training step."""
        sac = SAC(
            observation_space=continuous_env.observation_space,
            action_space=continuous_env.action_space,
            target_params=10_000,
            learning_starts=50,
            batch_size=32,
        )

        # Collect enough samples
        sac.collect_rollouts(continuous_env, n_steps=100)

        # Train
        metrics = sac.train()

        assert "critic1_loss" in metrics
        assert "actor_loss" in metrics


class TestDQN:
    """Tests for DQN algorithm."""

    def test_initialization(self, discrete_env):
        """Test DQN initialization."""
        dqn = DQN(
            observation_space=discrete_env.observation_space,
            action_space=discrete_env.action_space,
            target_params=10_000,
            buffer_size=1000,
        )

        assert dqn is not None
        assert dqn.q_network is not None
        assert dqn.target_network is not None

    def test_predict(self, discrete_env):
        """Test action prediction."""
        dqn = DQN(
            observation_space=discrete_env.observation_space,
            action_space=discrete_env.action_space,
            target_params=10_000,
        )

        obs, _ = discrete_env.reset()
        action = dqn.predict(obs, deterministic=True)

        assert 0 <= action < discrete_env.action_space.n

    def test_exploration_rate(self, discrete_env):
        """Test exploration rate schedule."""
        dqn = DQN(
            observation_space=discrete_env.observation_space,
            action_space=discrete_env.action_space,
            target_params=10_000,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.01,
            total_timesteps=10_000,
        )

        # Initial exploration rate
        assert dqn.exploration_rate == 1.0

    def test_collect_and_train(self, discrete_env):
        """Test collection and training."""
        dqn = DQN(
            observation_space=discrete_env.observation_space,
            action_space=discrete_env.action_space,
            target_params=10_000,
            learning_starts=50,
            batch_size=32,
        )

        dqn.collect_rollouts(discrete_env, n_steps=100)
        metrics = dqn.train()

        assert "loss" in metrics


class TestPPO:
    """Tests for PPO algorithm."""

    def test_initialization_discrete(self, discrete_env):
        """Test PPO initialization with discrete actions."""
        ppo = PPO(
            observation_space=discrete_env.observation_space,
            action_space=discrete_env.action_space,
            target_params=10_000,
            n_steps=64,
        )

        assert ppo is not None
        assert ppo.policy is not None
        assert ppo.value is not None

    def test_initialization_continuous(self, continuous_env):
        """Test PPO initialization with continuous actions."""
        ppo = PPO(
            observation_space=continuous_env.observation_space,
            action_space=continuous_env.action_space,
            target_params=10_000,
            n_steps=64,
        )

        assert ppo is not None

    def test_predict_discrete(self, discrete_env):
        """Test action prediction for discrete actions."""
        ppo = PPO(
            observation_space=discrete_env.observation_space,
            action_space=discrete_env.action_space,
            target_params=10_000,
        )

        obs, _ = discrete_env.reset()
        action = ppo.predict(obs, deterministic=True)

        assert 0 <= action < discrete_env.action_space.n

    def test_predict_continuous(self, continuous_env):
        """Test action prediction for continuous actions."""
        ppo = PPO(
            observation_space=continuous_env.observation_space,
            action_space=continuous_env.action_space,
            target_params=10_000,
        )

        obs, _ = continuous_env.reset()
        action = ppo.predict(obs, deterministic=True)

        assert action.shape == continuous_env.action_space.shape

    def test_collect_and_train(self, discrete_env):
        """Test rollout collection and training."""
        ppo = PPO(
            observation_space=discrete_env.observation_space,
            action_space=discrete_env.action_space,
            target_params=10_000,
            n_steps=64,
            n_epochs=2,
            batch_size=32,
        )

        ppo.collect_rollouts(discrete_env, n_steps=64)
        metrics = ppo.train()

        assert "pg_loss" in metrics
        assert "value_loss" in metrics

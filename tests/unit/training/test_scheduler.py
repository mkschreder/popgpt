"""Tests for learning rate scheduler."""

import math

import pytest

from popgpt.training import CosineDecayWithWarmup


class TestCosineDecayWithWarmup:
    """Tests for cosine decay with warmup scheduler."""

    def test_linear_warmup_phase(self) -> None:
        """Learning rate increases linearly during warmup."""
        # Arrange
        scheduler = CosineDecayWithWarmup(
            learning_rate=1.0,
            warmup_iters=100,
            lr_decay_iters=1000,
            min_lr=0.1,
        )

        # Act & Assert - should increase linearly
        lr_0 = scheduler.get_lr(0)
        lr_50 = scheduler.get_lr(50)
        lr_100 = scheduler.get_lr(100)

        assert lr_0 < lr_50 < lr_100
        assert math.isclose(lr_0, 1.0 * 1 / 101, rel_tol=1e-5)
        assert math.isclose(lr_50, 1.0 * 51 / 101, rel_tol=1e-5)

    def test_cosine_decay_phase(self) -> None:
        """Learning rate decays smoothly after warmup."""
        # Arrange
        scheduler = CosineDecayWithWarmup(
            learning_rate=1.0,
            warmup_iters=100,
            lr_decay_iters=1000,
            min_lr=0.1,
        )

        # Act
        lr_200 = scheduler.get_lr(200)
        lr_500 = scheduler.get_lr(500)
        lr_900 = scheduler.get_lr(900)

        # Assert - should decrease smoothly
        assert lr_200 > lr_500 > lr_900
        assert lr_900 > 0.1  # Above min_lr
        assert lr_200 < 1.0  # Below max

    def test_min_lr_after_decay_period(self) -> None:
        """Learning rate stays at min_lr after decay period ends."""
        # Arrange
        scheduler = CosineDecayWithWarmup(
            learning_rate=1.0,
            warmup_iters=100,
            lr_decay_iters=1000,
            min_lr=0.1,
        )

        # Act
        lr_1000 = scheduler.get_lr(1000)
        lr_1001 = scheduler.get_lr(1001)
        lr_2000 = scheduler.get_lr(2000)

        # Assert
        assert math.isclose(lr_1001, 0.1, rel_tol=1e-5)
        assert math.isclose(lr_2000, 0.1, rel_tol=1e-5)

    def test_constant_lr_when_decay_disabled(self) -> None:
        """Learning rate remains constant when decay_enabled=False."""
        # Arrange
        scheduler = CosineDecayWithWarmup(
            learning_rate=0.001,
            warmup_iters=100,
            lr_decay_iters=1000,
            min_lr=0.0001,
            decay_enabled=False,
        )

        # Act & Assert
        for iteration in [0, 50, 100, 500, 1000, 2000]:
            assert scheduler.get_lr(iteration) == 0.001

    @pytest.mark.parametrize(
        "iteration,expected_phase",
        [
            (0, "warmup"),
            (50, "warmup"),
            (99, "warmup"),
            (100, "warmup_end"),
            (101, "decay"),
            (500, "decay"),
            (999, "decay"),
            (1000, "min"),
            (2000, "min"),
        ],
        ids=[
            "start",
            "mid_warmup",
            "end_warmup",
            "warmup_boundary",
            "start_decay",
            "mid_decay",
            "end_decay",
            "at_min",
            "past_min",
        ],
    )
    def test_scheduler_phases(self, iteration: int, expected_phase: str) -> None:
        """Scheduler transitions correctly through phases."""
        scheduler = CosineDecayWithWarmup(
            learning_rate=1.0,
            warmup_iters=100,
            lr_decay_iters=1000,
            min_lr=0.1,
        )

        lr = scheduler.get_lr(iteration)

        if expected_phase == "warmup":
            # During warmup, should be less than max
            assert lr < 1.0
        elif expected_phase == "warmup_end":
            # At warmup end, close to max but still in warmup formula
            assert 0.9 < lr <= 1.0
        elif expected_phase == "decay":
            # During decay, between min and max
            assert 0.1 < lr < 1.0
        else:  # min
            # After decay, at min_lr
            assert math.isclose(lr, 0.1, rel_tol=1e-5)

    def test_zero_warmup(self) -> None:
        """Scheduler works correctly with warmup_iters=0."""
        scheduler = CosineDecayWithWarmup(
            learning_rate=1.0,
            warmup_iters=0,
            lr_decay_iters=1000,
            min_lr=0.1,
        )

        # Should start decay immediately
        lr_0 = scheduler.get_lr(0)
        assert 0.1 < lr_0 <= 1.0

    def test_realistic_gpt_schedule(self) -> None:
        """Scheduler works with realistic GPT training parameters."""
        # Arrange - typical GPT-2 schedule
        scheduler = CosineDecayWithWarmup(
            learning_rate=6e-4,
            warmup_iters=2000,
            lr_decay_iters=600000,
            min_lr=6e-5,
        )

        # Act & Assert key points
        lr_start = scheduler.get_lr(0)
        lr_warmup_end = scheduler.get_lr(2000)
        lr_mid = scheduler.get_lr(300000)
        lr_end = scheduler.get_lr(600000)
        lr_after = scheduler.get_lr(700000)

        assert lr_start < lr_warmup_end
        assert lr_mid > 6e-5
        assert lr_mid < 6e-4
        assert math.isclose(lr_after, 6e-5, rel_tol=1e-5)

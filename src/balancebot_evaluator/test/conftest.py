"""Pytest fixtures and configuration for BalanceBot E2E test suite."""

import pytest
from typing import Generator

from .evaluator_harness import (
    BalanceBotSimulator,
    FlightFSM,
    LegKinematics,
    LQRBalanceController,
    MetricsCalculator,
    RobotParams,
    SensorSuite,
    SkateparkEnvironment,
)


@pytest.fixture
def robot_params() -> RobotParams:
    """Fixture providing nominal robot physical parameters."""
    return RobotParams()


@pytest.fixture
def sim() -> Generator[BalanceBotSimulator, None, None]:
    """Fixture providing a fresh BalanceBot dynamic simulator."""
    simulator = BalanceBotSimulator()
    simulator.reset(x0=0.0, v0=0.0, theta0=0.0, height0=0.28)
    yield simulator


@pytest.fixture
def env() -> SkateparkEnvironment:
    """Fixture providing the skatepark terrain environment."""
    return SkateparkEnvironment()


@pytest.fixture
def kin(robot_params: RobotParams) -> LegKinematics:
    """Fixture providing 2-DOF leg kinematics solver."""
    return LegKinematics(robot_params)


@pytest.fixture
def lqr(robot_params: RobotParams) -> LQRBalanceController:
    """Fixture providing Gain-Scheduled LQR-I balance controller."""
    return LQRBalanceController(robot_params)


@pytest.fixture
def fsm() -> FlightFSM:
    """Fixture providing 5-state flight & landing state machine."""
    return FlightFSM()


@pytest.fixture
def sensors() -> SensorSuite:
    """Fixture providing sensor suite models."""
    return SensorSuite()


@pytest.fixture
def metrics_calc() -> MetricsCalculator:
    """Fixture providing quantitative metrics calculator."""
    return MetricsCalculator()

"""Algorithm registry for managing RL algorithms."""

from typing import Callable, Dict, Type

from rl_scaling_laws.algorithms.base import BaseAlgorithm


class AlgorithmRegistry:
    """Registry for RL algorithms."""

    _algorithms: Dict[str, Type[BaseAlgorithm]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register an algorithm.

        Args:
            name: Name to register the algorithm under.

        Returns:
            Decorator function.
        """
        def decorator(algo_cls: Type[BaseAlgorithm]) -> Type[BaseAlgorithm]:
            cls._algorithms[name.lower()] = algo_cls
            return algo_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseAlgorithm]:
        """Get an algorithm class by name.

        Args:
            name: Algorithm name.

        Returns:
            Algorithm class.

        Raises:
            KeyError: If algorithm not found.
        """
        if name.lower() not in cls._algorithms:
            available = ", ".join(cls._algorithms.keys())
            raise KeyError(f"Algorithm '{name}' not found. Available: {available}")
        return cls._algorithms[name.lower()]

    @classmethod
    def list_algorithms(cls) -> list:
        """List all registered algorithms."""
        return list(cls._algorithms.keys())


def register_algorithm(name: str) -> Callable:
    """Convenience function to register an algorithm."""
    return AlgorithmRegistry.register(name)


def get_algorithm(name: str, **kwargs) -> BaseAlgorithm:
    """Create an algorithm instance by name.

    Args:
        name: Algorithm name.
        **kwargs: Arguments for algorithm constructor.

    Returns:
        Algorithm instance.
    """
    algo_cls = AlgorithmRegistry.get(name)
    return algo_cls(**kwargs)

"""Model registry for architecture management."""

from typing import Any, Callable, Dict, Optional, Type

from rl_scaling_laws.models.base import BaseNetwork


class ModelRegistry:
    """Registry for model architectures."""

    _models: Dict[str, Type[BaseNetwork]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a model class.

        Args:
            name: Name to register the model under.

        Returns:
            Decorator function.
        """
        def decorator(model_cls: Type[BaseNetwork]) -> Type[BaseNetwork]:
            cls._models[name.lower()] = model_cls
            return model_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseNetwork]:
        """Get a model class by name.

        Args:
            name: Model name.

        Returns:
            Model class.

        Raises:
            KeyError: If model not found.
        """
        if name.lower() not in cls._models:
            available = ", ".join(cls._models.keys())
            raise KeyError(f"Model '{name}' not found. Available: {available}")
        return cls._models[name.lower()]

    @classmethod
    def list_models(cls) -> list:
        """List all registered models."""
        return list(cls._models.keys())


def register_model(name: str) -> Callable:
    """Convenience function to register a model.

    Args:
        name: Model name.

    Returns:
        Decorator function.
    """
    return ModelRegistry.register(name)


def get_model(name: str, **kwargs) -> BaseNetwork:
    """Create a model instance by name.

    Args:
        name: Model name.
        **kwargs: Arguments to pass to model constructor.

    Returns:
        Model instance.
    """
    model_cls = ModelRegistry.get(name)
    return model_cls(**kwargs)

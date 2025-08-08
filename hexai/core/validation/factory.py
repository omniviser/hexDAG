"""Validator factory for creating validators with different strategies."""

from .core import BaseValidator, IValidator
from .strategies import ValidationStrategy


class ValidatorFactory:
    """Factory for creating validators with specified strategies.

    Provides convenient methods to create validators without needing to import and instantiate
    validator classes directly.
    """

    @staticmethod
    def create(strategy: ValidationStrategy) -> IValidator:
        """Create a validator with the specified strategy.

        Parameters
        ----------
        strategy : ValidationStrategy
            The validation strategy to use

        Returns
        -------
        IValidator
            A validator instance configured with the specified strategy
        """
        return BaseValidator(strategy)

    @staticmethod
    def strict() -> IValidator:
        """Create a strict validator that fails fast on validation errors.

        Returns
        -------
        IValidator
            A validator with strict validation strategy
        """
        return BaseValidator(ValidationStrategy.STRICT)

    @staticmethod
    def coerce() -> IValidator:
        """Create a coercing validator that attempts type conversion.

        Returns
        -------
        IValidator
            A validator with coerce validation strategy
        """
        return BaseValidator(ValidationStrategy.COERCE)

    @staticmethod
    def passthrough() -> IValidator:
        """Create a passthrough validator that allows all data.

        Returns
        -------
        IValidator
            A validator with passthrough validation strategy
        """
        return BaseValidator(ValidationStrategy.PASSTHROUGH)


# Convenience functions for quick validator creation
def strict_validator() -> IValidator:
    """Create a strict validator.

    Returns
    -------
    IValidator
        A validator with strict validation strategy
    """
    return ValidatorFactory.strict()


def coerce_validator() -> IValidator:
    """Create a coercing validator.

    Returns
    -------
    IValidator
        A validator with coerce validation strategy
    """
    return ValidatorFactory.coerce()


def passthrough_validator() -> IValidator:
    """Create a passthrough validator.

    Returns
    -------
    IValidator
        A validator with passthrough validation strategy
    """
    return ValidatorFactory.passthrough()

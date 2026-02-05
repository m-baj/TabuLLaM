"""Parameter validation utilities."""

from ..exceptions import ConfigurationError


VALID_MODES = ['zero_shot', 'random_few_shot', 'semantic_few_shot']
VALID_PROMPT_TYPES = ['standard', 'binary_confidence', 'multiclass_confidence']
VALID_PROVIDERS = ['openai', 'ollama', 'google']


def validate_mode(mode: str) -> None:
    """
    Validate classification mode parameter.

    Parameters
    ----------
    mode : str
        Classification mode

    Raises
    ------
    ConfigurationError
        If mode is not valid
    """
    if mode not in VALID_MODES:
        raise ConfigurationError(
            f"Invalid mode: '{mode}'. Must be one of: {VALID_MODES}"
        )


def validate_prompt_type(prompt_type: str) -> None:
    """
    Validate prompt type parameter.

    Parameters
    ----------
    prompt_type : str
        Prompt type

    Raises
    ------
    ConfigurationError
        If prompt_type is not valid
    """
    if prompt_type not in VALID_PROMPT_TYPES:
        raise ConfigurationError(
            f"Invalid prompt_type: '{prompt_type}'. Must be one of: {VALID_PROMPT_TYPES}"
        )


def validate_llm_spec(llm_spec: str) -> tuple:
    """
    Validate and parse LLM specification string.

    Parameters
    ----------
    llm_spec : str
        LLM specification in format 'provider:model_name'

    Returns
    -------
    tuple
        (provider, model_name)

    Raises
    ------
    ConfigurationError
        If llm_spec format is invalid
    """
    if ':' not in llm_spec:
        raise ConfigurationError(
            f"Invalid llm format: '{llm_spec}'. Expected format: 'provider:model_name'"
        )

    provider, model = llm_spec.split(':', 1)

    if provider not in VALID_PROVIDERS:
        raise ConfigurationError(
            f"Invalid provider: '{provider}'. Must be one of: {VALID_PROVIDERS}"
        )

    return provider, model

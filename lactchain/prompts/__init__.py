"""PromptTemplate module."""

from __future__ import annotations

from typing import Any
from typing import Union

from lactchain.classes.base_prompt import BasePromptTemplate
from lactchain.prompts.actions import ActionsPromptTemplateConfig, ActionPromptTemplate, ActionSolver
from lactchain.prompts.strategies import StrategyPromptTemplateConfig, StrategyPromptTemplate
from lactchain.prompts.critic import GridWorldPromptTemplateConfig, GridWorldOutput

PromptTemplateConfigs = Union[
    ActionsPromptTemplateConfig,
    StrategyPromptTemplateConfig, 
    GridWorldPromptTemplateConfig
]

PROMPT_TEMPLATES = {
    'action': (ActionsPromptTemplateConfig, ActionPromptTemplate),
    'strategy': (StrategyPromptTemplateConfig, StrategyPromptTemplate),
    'critic': (GridWorldPromptTemplateConfig, GridWorldOutput)
}

ACTION_SOLVERS = {
    'gridworld': ActionSolver
}

def get_prompt_template(config: PromptTemplateConfigs) -> BasePromptTemplate:
    """Get the instance based on the kwargs.

    Currently supports the following templates: 
    - Action
    - Strategy

    Parameters
    ----------
    kwargs : dict[str, Any]
        The configuration. Contains a `name` argument
        to specify the strategy to use.

    Returns
    -------
    PromptTemplate
        The instance.

    Raises
    ------
    ValueError
        If the `name` is unknown.
    """
    
    kwargs = config.model_dump()
    
    name = kwargs.get('name', '')
    strategy = PROMPT_TEMPLATES.get(name)
    if not strategy:
        raise ValueError(
            f'Unknown prompt name: {name}.'
            f' Available: {set(PROMPT_TEMPLATES.keys())}',
        )

    config_cls, cls = strategy
    return cls(config_cls(**kwargs))


def get_action_solver(config: PromptTemplateConfigs) -> ActionSolver: 
    '''Grabs solver that maps strings to encoded actions compatible with the environment'''
    
    kwargs = config.model_dump()
    
    solver = kwargs.get('solver', '')
    strategy = ACTION_SOLVERS.get(solver)
    if not strategy:
        raise ValueError(
            f'Unknown prompt name: {solver}.'
            f' Available: {set(ACTION_SOLVERS.keys())}',
        )

    cls = strategy()
    return cls
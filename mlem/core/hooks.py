"""
Base classes for Hook and Analyzer.
Hook identifies whether the object matches the hook and processes it.
Analyzer keeps track of all imported hooks and applies them to the object
    to find suitable one.
"""
import inspect
import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, List, Tuple, Type, TypeVar

from mlem.core.errors import HookNotFound, MultipleHooksFound

logger = logging.getLogger(__name__)

ANALYZER_FIELD = "analyzer"

TOP_PRIORITY_VALUE = "top"
LOW_PRIORITY_VALUE = "low"

T = TypeVar("T")


class Hook(ABC, Generic[T]):
    """
    Base class for Hooks.
    Hooks are used by corresponding `Analyzer` to analyze and process objects
    """

    priority = 0
    analyzer: Type["Analyzer"]

    @classmethod
    def get_priority(cls):
        return cls.priority

    @classmethod
    @abstractmethod
    def is_object_valid(cls, obj: Any) -> bool:
        """
        Must return True if obj can be processed by this hook

        :param obj: object to analyze
        :return: True or False
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def process(cls, obj: Any, **kwargs) -> T:
        """
        Analyzes obj and returns result. Result type is determined by specific Hook class sub-hierarchy

        :param obj: object to analyze
        :param kwargs: additional information to be used for analysis
        :return: analysis result
        """
        raise NotImplementedError

    def __init_subclass__(cls, *args, **kwargs):
        if not inspect.isabstract(cls):
            analyzer = cls.analyzer
            if analyzer is not None:
                analyzer.hooks.append(cls)
                logger.debug(
                    "Registering %s to %s", cls.__name__, analyzer.__name__
                )
        else:
            logger.debug(
                "Not registerting %s to any Analyzer because it's an abstract class",
                cls.__name__,
            )
        super(Hook, cls).__init_subclass__(*args, **kwargs)


class IsInstanceHookMixin(Hook, ABC):
    valid_types: ClassVar[Tuple[Type, ...]]

    @classmethod
    def is_object_valid(cls, obj: Any) -> bool:
        return isinstance(obj, cls.valid_types)


class Analyzer(Generic[T]):
    """Base class for analyzers.
    Analyzers hold list of corresponding hooks. The `analyze` method goes through
    all the hook to find valid one and uses it to `process` the object"""

    base_hook_class: Type[Hook[T]]

    hooks: List[Type[Hook[T]]]

    def __init_subclass__(cls, *args, **kwargs):
        cls.base_hook_class.analyzer = cls
        cls.hooks = []
        super(Analyzer, cls).__init_subclass__(*args, **kwargs)

    @classmethod
    def analyze(cls, obj, **kwargs) -> T:
        """
        Run this analyzer's base_hook_class hooks to analyze obj

        :param obj: objects to analyze
        :param kwargs: additional information to be used for analysis
        :return: Result of processing obj with base_hook_class subtype
        """
        return cls._find_hook(obj).process(obj, **kwargs)

    @classmethod
    def _find_hook(cls, obj) -> Type[Hook[T]]:
        hooks = []
        lp_hooks = []
        for hook in cls.hooks:
            if hook.is_object_valid(obj):
                hook_priority = hook.get_priority()
                if hook_priority == TOP_PRIORITY_VALUE:
                    logger.debug(
                        "processing class %s with %s",
                        type(obj).__name__,
                        hook.__class__.__name__,
                    )
                    return hook
                if hook_priority == LOW_PRIORITY_VALUE:
                    lp_hooks.append(hook)
                else:
                    hooks.append((hook_priority, hook))

        if len(hooks) == 0:
            if len(lp_hooks) == 1:
                return lp_hooks[0]
            if len(lp_hooks) > 1:
                raise MultipleHooksFound(
                    f"Multiple suitable hooks for object {obj} ({lp_hooks})"
                )
            raise HookNotFound(
                f"No suitable {cls.base_hook_class.__name__} for object of type "
                f'"{type(obj).__name__}". Registered hooks: {cls.hooks}'
            )
        return max(hooks, key=lambda x: x[0])[1]

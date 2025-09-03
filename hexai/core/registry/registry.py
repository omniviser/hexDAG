"""Singleton component registry for hexDAG.

This module provides a centralized registry for all hexDAG components,
supporting plugins while maintaining control over component registration.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Any, Callable

from hexai.core.registry.decorators import get_component_metadata
from hexai.core.registry.discovery import discover_entry_points
from hexai.core.registry.metadata import ComponentMetadata
from hexai.core.registry.types import ComponentType

logger = logging.getLogger(__name__)

# Module-level lock for thread-safe singleton initialization
# This avoids the race condition where class-level _lock could be accessed
# before the class is fully initialized in multi-threaded environments
_singleton_lock = threading.RLock()


class ComponentRegistry:
    """Singleton append-only registry for all hexDAG components.

    Provides centralized component registration and discovery with support
    for plugins and namespaced components. Similar to Django's apps registry
    but designed for diverse component types rather than just models.

    The registry follows an append-only pattern for production stability:
    - Components can be registered but not removed
    - Replacement is allowed with explicit `replace=True` flag and warnings
    - Ensures components dependencies remain stable at runtime

    Examples
    --------
    >>> from hexai.core.registry import registry
    >>>
    >>> # Register a component
    >>> registry.register('my_node', MyNode(), component_type=ComponentType.NODE)
    >>>
    >>> # Get a component
    >>> node = registry.get('my_node', component_type=ComponentType.NODE)
    >>>
    >>> # List all nodes
    >>> nodes = registry.list_components(component_type=ComponentType.NODE)
    >>>
    >>> # Plugin registration
    >>> registry.register_namespace('my_plugin')
    >>> registry.register('analyzer', Analyzer(),
    ...                   component_type=ComponentType.NODE,
    ...                   namespace='my_plugin')
    >>>
    >>> # Mark component as replaceable during registration
    >>> metadata = ComponentMetadata(
    ...     name='my_node',
    ...     component_type=ComponentType.NODE,
    ...     replaceable=True  # Allow future replacements
    ... )
    >>> registry.register('my_node', MyNode(),
    ...                   component_type=ComponentType.NODE,
    ...                   metadata=metadata)
    >>>
    >>> # Now can replace it
    >>> registry.register('my_node', ImprovedNode(),
    ...                   component_type=ComponentType.NODE,
    ...                   replace=True)  # Works because replaceable=True
    """

    _instance: ComponentRegistry | None = None

    def __new__(cls) -> ComponentRegistry:
        """Ensure only one instance exists (singleton pattern).

        Uses a module-level lock instead of class-level to avoid race conditions
        where the class variable _lock could be accessed before initialization.
        This is a more robust pattern for thread-safe singletons.
        """
        # Fast path: if instance exists, return it (no lock needed)
        if cls._instance is not None:
            return cls._instance

        # Slow path: acquire lock and create instance if needed
        with _singleton_lock:
            # Double-check after acquiring lock
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the registry (only runs once due to singleton).

        Thread-safe initialization using the module-level lock.
        """
        # Use module lock for thread-safe initialization check
        with _singleton_lock:
            if hasattr(self, "_initialized"):
                return

            # Create instance-level lock for normal operations
            self._lock = threading.RLock()

            # Component storage: {component_type: {namespace: {name: (metadata, component)}}}
            self._components: dict[str, dict[str, dict[str, tuple[ComponentMetadata, Any]]]] = (
                defaultdict(lambda: defaultdict(dict))
            )

            # Protected system namespaces with allowed packages (like Django apps)
            self._protected_namespaces: dict[str, set[str]] = {
                "core": {"hexai.core"},  # Core hexDAG components
                "system": {"hexai.core", "hexai.system"},  # System-level components
                "internal": {"hexai.core", "hexai.internal"},  # Internal implementation details
                "hexai": {"hexai"},  # First-party hexai extensions
            }

            # Registered namespaces
            self._namespaces: set[str] = set(self._protected_namespaces.keys())

            # Registration hooks for plugins
            self._pre_register_hooks: list[Callable] = []
            self._post_register_hooks: list[Callable] = []

            # Dependency graph for resolution
            self._dependency_graph: dict[str, set[str]] = defaultdict(set)

            # Reverse dependency tracking (who depends on this component)
            self._dependents_graph: dict[str, set[str]] = defaultdict(set)

            # Registry state
            self._ready = False
            self._plugins_loaded = False

            # Lazy loading state for plugins
            self._available_plugins: dict[str, Any] = {}  # namespace -> entry_point
            self._loaded_plugins: set[str] = set()  # Track which plugins are loaded

            # Mark as initialized BEFORE loading components to prevent recursion
            self._initialized = True

        # Load components outside the singleton lock to avoid deadlocks
        # These operations use the instance-level _lock internally
        self._load_core_components()
        self._register_pending_components()

        logger.info("Component registry initialized")

    def register(
        self,
        name: str,
        component: Any,
        component_type: ComponentType | str,
        namespace: str,
        metadata: ComponentMetadata | None = None,
        replace: bool = False,
        force_replace: bool = False,
        **kwargs: Any,
    ) -> None:
        """Register a component in the registry.

        Parameters
        ----------
        name : str
            Unique identifier for the component within its namespace.
        component : Any
            The component instance to register.
        component_type : ComponentTypeEnum
            Type of component (node, adapter, tool, etc.).
        metadata : ComponentMetadata | None
            Optional metadata. If not provided, will be created.
        namespace : str
            Required namespace for the component.
            Use 'hexai' for first-party components or your plugin name for third-party.
        replace : bool
            If True, replace existing component if it's marked as replaceable
            and has no dependents. Otherwise raise error.
        force_replace : bool
            If True, force replacement even if component has dependents.
            Use with extreme caution as it may break dependent components.
        **kwargs : Any
            Additional metadata fields.

        Raises
        ------
        ValueError
            If namespace not registered or component already exists (and replace=False).

        Examples
        --------
        >>> registry.register('llm_node', LLMNode(), ComponentType.NODE)
        >>>
        >>> # Plugin registration
        >>> registry.register_namespace('my_plugin')
        >>> registry.register('custom_node', CustomNode(), ComponentType.NODE,
        ...                  namespace='my_plugin')
        """
        with self._lock:
            # Validate namespace protection (similar to Django's app registry)
            if namespace in self._protected_namespaces:
                allowed_packages = self._protected_namespaces[namespace]
                if not self._is_caller_allowed(allowed_packages):
                    raise ValueError(
                        f"Namespace '{namespace}' is protected and reserved for system use. "
                        f"Allowed packages: {', '.join(allowed_packages)}. "
                        f"Please use your own namespace (e.g., your plugin name)."
                    )

            # Auto-create namespace if it doesn't exist (for non-protected namespaces)
            if namespace not in self._namespaces:
                logger.info(
                    "Auto-registering namespace '%s'. "
                    "Consider explicitly registering with register_namespace() for clarity.",
                    namespace,
                )
                self._namespaces.add(namespace)

            # Check for existing component
            type_registry = self._components[component_type]
            namespace_registry = type_registry[namespace]

            if name in namespace_registry:
                existing_metadata, _ = namespace_registry[name]
                full_name = f"{namespace}:{name}"

                # Check if any components depend on this one
                dependents = self._dependents_graph.get(full_name, set())

                # Check replacement conditions
                if not existing_metadata.replaceable and not replace:
                    raise ValueError(
                        f"Component '{namespace}:{name}' is not replaceable. "
                        f"It was registered with replaceable=False for stability."
                    )
                elif not existing_metadata.replaceable and replace:
                    raise ValueError(
                        f"Component '{namespace}:{name}' is protected and cannot be replaced. "
                        f"The component was explicitly marked as non-replaceable."
                    )
                elif dependents and replace and not force_replace:
                    # Component has dependents, block replacement unless forced
                    dependent_names = ", ".join(sorted(dependents)[:3])
                    more = f" and {len(dependents) - 3} more" if len(dependents) > 3 else ""
                    raise ValueError(
                        f"Component '{namespace}:{name}' cannot be replaced because it has dependents: "
                        f"{dependent_names}{more}. "
                        f"Replacing it could break these components. "
                        f"Use force_replace=True only if all dependents are compatible."
                    )
                elif dependents and replace and force_replace:
                    logger.error(
                        "FORCE REPLACING component '%s:%s' despite having %d dependents: %s. "
                        "This may break dependent components!",
                        namespace,
                        name,
                        len(dependents),
                        ", ".join(sorted(dependents)),
                    )
                elif existing_metadata.replaceable and not replace:
                    raise ValueError(
                        f"Component '{namespace}:{name}' already registered. "
                        f"Use replace=True to override (component is marked as replaceable)."
                    )
                else:  # existing_metadata.replaceable and replace and no dependents
                    logger.warning(
                        "Replacing component '%s:%s' (marked as replaceable, no dependents). "
                        "Ensure any future components are compatible.",
                        namespace,
                        name,
                    )

            # Create metadata if needed (or use decorated metadata)
            if metadata is None:
                # Check if component has decorated metadata
                decorated_metadata = get_component_metadata(component)

                if decorated_metadata:
                    # Use decorated metadata, but override name and add kwargs
                    metadata = ComponentMetadata(
                        name=name,
                        component_type=component_type,
                        description=decorated_metadata.description,
                        tags=decorated_metadata.tags,
                        author=decorated_metadata.author,
                        dependencies=decorated_metadata.dependencies,
                        config_schema=decorated_metadata.config_schema,
                        replaceable=kwargs.pop("replaceable", decorated_metadata.replaceable),
                        **kwargs,
                    )
                else:
                    # Create fresh metadata
                    replaceable = kwargs.pop("replaceable", False)
                    metadata = ComponentMetadata(
                        name=name, component_type=component_type, replaceable=replaceable, **kwargs
                    )

            # Run pre-registration hooks
            for hook in self._pre_register_hooks:
                hook(name, component, metadata, namespace)

            # Register component
            namespace_registry[name] = (metadata, component)

            # Update dependency graph
            full_name = f"{namespace}:{name}"
            if metadata.dependencies:
                deps = {
                    dep if ":" in dep else f"{namespace}:{dep}" for dep in metadata.dependencies
                }
                self._dependency_graph[full_name] = deps

                # Update reverse dependencies (track who depends on each component)
                for dep in deps:
                    self._dependents_graph[dep].add(full_name)

            # Run post-registration hooks
            for hook in self._post_register_hooks:
                hook(name, component, metadata, namespace)

            logger.debug(
                "Registered component '%s:%s' of type '%s'", namespace, name, component_type
            )

    def _ensure_plugin_loaded(self, namespace: str) -> None:
        """Ensure a plugin namespace is loaded (lazy loading).

        Parameters
        ----------
        namespace : str
            Namespace to check and load if needed.
        """
        # Skip if not a plugin namespace or already loaded
        if namespace in self._protected_namespaces:
            return
        if namespace in self._loaded_plugins:
            return

        # Check if this namespace has an available plugin
        if namespace in self._available_plugins:
            logger.debug(f"Lazy loading plugin for namespace '{namespace}'")
            try:
                # Get the plugin's register function
                register_func = self._available_plugins[namespace]

                # Call the register function to load the plugin
                register_func()

                # Mark as loaded
                self._loaded_plugins.add(namespace)
                logger.info(f"Lazy loaded plugin '{namespace}'")
            except Exception as e:
                logger.error(f"Failed to lazy load plugin '{namespace}': {e}")

    def get(
        self, name: str, component_type: ComponentType | str | None = None, namespace: str = "core"
    ) -> Any:
        """Get a registered component.

        Parameters
        ----------
        name : str
            Component name. Can include namespace as 'namespace:name'.
        component_type : ComponentTypeEnum | None
            Component type. If None, searches all types.
        namespace : str
            Namespace to search in (default: 'core').

        Returns
        -------
        Any
            The registered component.

        Raises
        ------
        KeyError
            If component not found.

        Examples
        --------
        >>> node = registry.get('my_node', ComponentType.NODE)
        >>>
        >>> # With namespace prefix
        >>> plugin_node = registry.get('my_plugin:custom_node', ComponentType.NODE)
        >>>
        >>> # Search all types
        >>> component = registry.get('some_component')
        """
        with self._lock:
            # Parse namespace from name if provided
            if ":" in name:
                namespace, name = name.split(":", 1)

            # Ensure plugin is loaded if accessing plugin namespace (lazy loading)
            self._ensure_plugin_loaded(namespace)

            # If type specified, direct lookup
            if component_type:
                type_registry = self._components.get(component_type, {})
                namespace_registry = type_registry.get(namespace, {})

                if name not in namespace_registry:
                    raise KeyError(
                        f"Component '{namespace}:{name}' not found in {component_type}. "
                        f"Available: {list(namespace_registry.keys())}"
                    )

                _, component = namespace_registry[name]
                return component

            # Search all types
            for comp_type in self._components:
                namespace_registry = self._components[comp_type].get(namespace, {})
                if name in namespace_registry:
                    _, component = namespace_registry[name]
                    return component

            raise KeyError(f"Component '{namespace}:{name}' not found in any type")

    def get_metadata(
        self, name: str, component_type: ComponentType | str | None = None, namespace: str = "core"
    ) -> ComponentMetadata:
        """Get metadata for a registered component.

        Parameters
        ----------
        name : str
            Component name. Can include namespace as 'namespace:name'.
        component_type : ComponentTypeEnum | None
            Component type. If None, searches all types.
        namespace : str
            Namespace to search in (default: 'core').

        Returns
        -------
        ComponentMetadata
            The component's metadata.

        Raises
        ------
        KeyError
            If component not found.
        """
        with self._lock:
            # Parse namespace from name if provided
            if ":" in name:
                namespace, name = name.split(":", 1)

            # Ensure plugin is loaded if accessing plugin namespace (lazy loading)
            self._ensure_plugin_loaded(namespace)

            # If type specified, direct lookup
            if component_type:
                type_registry = self._components.get(component_type, {})
                namespace_registry = type_registry.get(namespace, {})

                if name not in namespace_registry:
                    raise KeyError(f"Component '{namespace}:{name}' not found")

                metadata, _ = namespace_registry[name]
                return metadata

            # Search all types
            for comp_type in self._components:
                namespace_registry = self._components[comp_type].get(namespace, {})
                if name in namespace_registry:
                    metadata, _ = namespace_registry[name]
                    return metadata

            raise KeyError(f"Component '{namespace}:{name}' not found")

    def list_components(
        self,
        component_type: ComponentType | str | None = None,
        namespace: str | None = None,
        include_metadata: bool = False,
        include_unloaded: bool = False,
    ) -> list[str] | list[tuple[str, ComponentMetadata]]:
        """List registered components.

        Parameters
        ----------
        component_type : ComponentTypeEnum | None
            Filter by component type. If None, returns all types.
        namespace : str | None
            Filter by namespace. If None, returns all namespaces.
        include_metadata : bool
            If True, return tuples of (name, metadata).
        include_unloaded : bool
            If True, also return components from discovered but unloaded plugins.

        Returns
        -------
        list[str] | list[tuple[str, ComponentMetadata]]
            List of component names or (name, metadata) tuples.

        Examples
        --------
        >>> # List all nodes
        >>> nodes = registry.list_components(component_type=ComponentType.NODE)
        >>>
        >>> # List plugin components
        >>> plugin_components = registry.list_components(namespace='my_plugin')
        >>>
        >>> # Get all components with metadata
        >>> all_components = registry.list_components(include_metadata=True)
        """
        with self._lock:
            results: list[Any] = []

            # Determine which types to search
            types_to_search = [component_type] if component_type else list(self._components.keys())

            # Determine which namespaces to search
            for comp_type in types_to_search:
                type_registry = self._components.get(comp_type, {})

                namespaces_to_search = [namespace] if namespace else list(type_registry.keys())

                # Load plugins if needed (unless include_unloaded is True)
                if not include_unloaded:
                    for ns in namespaces_to_search:
                        self._ensure_plugin_loaded(ns)

                for ns in namespaces_to_search:
                    namespace_registry = type_registry.get(ns, {})

                    for name, (metadata, _) in namespace_registry.items():
                        full_name = f"{ns}:{name}" if ns != "core" else name

                        if include_metadata:
                            results.append((full_name, metadata))
                        else:
                            results.append(full_name)

            return sorted(results, key=lambda x: x[0] if isinstance(x, tuple) else x)

    def find(self, **filters: Any) -> dict[str, ComponentMetadata]:
        """Find components matching filters.

        Parameters
        ----------
        **filters : Any
            Filter criteria. Supports special operations:
            - field__contains: Check if value is in collection
            - field__startswith: String prefix matching
            - field__in: Check if field in provided collection

        Returns
        -------
        dict[str, ComponentMetadata]
            Mapping of component names to metadata.

        Examples
        --------
        >>> # Find all LLM-related components
        >>> llm_components = registry.find(tags__contains='llm')
        >>>
        >>> # Find by author
        >>> my_components = registry.find(author='hexdag')
        """
        with self._lock:
            results = {}

            for comp_type in self._components:
                for namespace in self._components[comp_type]:
                    for name, (metadata, _) in self._components[comp_type][namespace].items():
                        if self._matches_filters(metadata, filters):
                            full_name = f"{namespace}:{name}" if namespace != "core" else name
                            results[full_name] = metadata

            return results

    def _matches_filters(self, metadata: ComponentMetadata, filters: dict[str, Any]) -> bool:
        """Check if metadata matches all filters."""
        for filter_key, filter_value in filters.items():
            if "__" in filter_key:
                field_name, operation = filter_key.split("__", 1)
                field_value = getattr(metadata, field_name, None)

                if not self._apply_filter_operation(field_value, operation, filter_value):
                    return False
            else:
                field_value = getattr(metadata, filter_key, None)
                if field_value != filter_value:
                    return False

        return True

    def _apply_filter_operation(self, field_value: Any, operation: str, filter_value: Any) -> bool:
        """Apply a filter operation to a field value."""
        match operation:
            case "contains":
                if isinstance(field_value, (set, frozenset, list, tuple)):
                    return filter_value in field_value
                if isinstance(field_value, str):
                    return filter_value in field_value
                return False
            case "startswith":
                if isinstance(field_value, str):
                    return field_value.startswith(filter_value)
                return False
            case "in":
                return field_value in filter_value
            case _:
                logger.warning(f"Unsupported filter operation: {operation}")
                return False

    def register_namespace(self, namespace: str) -> None:
        """Register a new namespace for plugins.

        Parameters
        ----------
        namespace : str
            Namespace identifier. Should be unique per plugin.
            Cannot use protected namespaces (core, system, internal).

        Raises
        ------
        ValueError
            If namespace already registered or is protected.

        Examples
        --------
        >>> # Plugin registers its namespace
        >>> registry.register_namespace('my_awesome_plugin')
        >>>
        >>> # Now can register components in that namespace
        >>> registry.register('tool', MyTool(), ComponentType.TOOL,
        ...                  namespace='my_awesome_plugin')
        """
        with self._lock:
            if namespace in self._protected_namespaces:
                raise ValueError(
                    f"Namespace '{namespace}' is protected and cannot be registered. "
                    f"Protected namespaces: {sorted(self._protected_namespaces)}"
                )

            if namespace in self._namespaces:
                logger.debug(f"Namespace '{namespace}' already registered")
                return  # Idempotent - don't error if already registered

            self._namespaces.add(namespace)
            logger.info(f"Registered namespace '{namespace}'")

    def _unregister_namespace_unsafe(self, namespace: str) -> None:
        """[INTERNAL USE ONLY] Unregister a namespace and all its components.

        This method is preserved for testing purposes only and should not be
        used in production code. The registry follows an append-only pattern
        to ensure system stability.

        Parameters
        ----------
        namespace : str
            Namespace to unregister.

        Raises
        ------
        ValueError
            If trying to unregister 'core' namespace.
        DeprecationWarning
            Always raised to discourage use.
        """
        import warnings

        warnings.warn(
            "Unregistering namespaces is deprecated and will be removed. "
            "The registry should be append-only in production.",
            DeprecationWarning,
            stacklevel=2,
        )

        with self._lock:
            if namespace == "core":
                raise ValueError("Cannot unregister 'core' namespace")

            if namespace not in self._namespaces:
                raise ValueError(f"Namespace '{namespace}' not registered")

            # Remove all components in namespace
            for comp_type in self._components:
                if namespace in self._components[comp_type]:
                    count = len(self._components[comp_type][namespace])
                    del self._components[comp_type][namespace]
                    logger.warning(
                        f"[UNSAFE] Removed {count} {comp_type} components from namespace '{namespace}'"
                    )

            # Remove namespace
            self._namespaces.remove(namespace)
            logger.warning(f"[UNSAFE] Unregistered namespace '{namespace}'")

    def list_namespaces(self) -> list[str]:
        """List all registered namespaces.

        Returns
        -------
        list[str]
            Sorted list of namespace names.
        """
        with self._lock:
            return sorted(self._namespaces)

    def list_available_plugins(self) -> dict[str, bool]:
        """List discovered plugins and their load status.

        Returns
        -------
        dict[str, bool]
            Mapping of plugin namespace to loaded status.

        Examples
        --------
        >>> plugins = registry.list_available_plugins()
        >>> for namespace, is_loaded in plugins.items():
        ...     status = "loaded" if is_loaded else "available"
        ...     print(f"{namespace}: {status}")
        """
        with self._lock:
            return {
                namespace: namespace in self._loaded_plugins
                for namespace in self._available_plugins
            }

    def load_plugin(self, namespace: str) -> bool:
        """Explicitly load a plugin by namespace.

        Parameters
        ----------
        namespace : str
            Plugin namespace to load.

        Returns
        -------
        bool
            True if plugin was loaded successfully.

        Examples
        --------
        >>> # Explicitly load a plugin
        >>> if registry.load_plugin('nlp_tools'):
        ...     print("NLP tools plugin loaded")
        """
        with self._lock:
            if namespace not in self._available_plugins:
                logger.error(f"Plugin '{namespace}' not found in available plugins")
                return False

            if namespace in self._loaded_plugins:
                logger.debug(f"Plugin '{namespace}' already loaded")
                return True

            self._ensure_plugin_loaded(namespace)
            return namespace in self._loaded_plugins

    def add_hook(self, hook: Callable, hook_type: str = "pre") -> None:
        """Add a registration hook.

        Parameters
        ----------
        hook : Callable
            Hook function with signature (name, component, metadata, namespace).
        hook_type : str
            Either 'pre' or 'post' for pre/post registration hooks.

        Examples
        --------
        >>> def validate_component(name, component, metadata, namespace):
        ...     if not hasattr(component, 'execute'):
        ...         raise ValueError(f"Component {name} must have execute method")
        >>>
        >>> registry.add_hook(validate_component, 'pre')
        """
        with self._lock:
            if hook_type == "pre":
                self._pre_register_hooks.append(hook)
            elif hook_type == "post":
                self._post_register_hooks.append(hook)
            else:
                raise ValueError(f"Invalid hook type: {hook_type}")

    def remove_hook(self, hook: Callable, hook_type: str = "pre") -> None:
        """Remove a registration hook.

        Parameters
        ----------
        hook : Callable
            Hook function to remove.
        hook_type : str
            Either 'pre' or 'post'.
        """
        with self._lock:
            if hook_type == "pre":
                self._pre_register_hooks.remove(hook)
            elif hook_type == "post":
                self._post_register_hooks.remove(hook)

    def load_plugins(self, entry_point_group: str = "hexdag.plugins") -> int:
        """Load plugins from entry points.

        Parameters
        ----------
        entry_point_group : str
            Entry point group to load from.

        Returns
        -------
        int
            Number of plugins loaded.

        Examples
        --------
        >>> # In plugin's setup.py:
        >>> # entry_points={
        >>> #     'hexdag.plugins': [
        >>> #         'my_plugin = my_plugin:register'
        >>> #     ]
        >>> # }
        >>>
        >>> # Load all plugins
        >>> count = registry.load_plugins()
        >>> print(f"Loaded {count} plugins")
        """
        with self._lock:
            if self._plugins_loaded:
                logger.warning("Plugins already loaded")
                return 0

            plugins = discover_entry_points(entry_point_group)
            loaded = 0

            for plugin_name, plugin_register in plugins.items():
                try:
                    # Register plugin namespace
                    self.register_namespace(plugin_name)

                    # Call plugin's register function
                    if callable(plugin_register):
                        plugin_register(self, plugin_name)

                    loaded += 1
                    logger.info(f"Loaded plugin '{plugin_name}'")

                except Exception as e:
                    logger.error(f"Failed to load plugin '{plugin_name}': {e}")

            self._plugins_loaded = True
            return loaded

    def _is_caller_allowed(self, allowed_packages: set[str]) -> bool:
        """Check if the calling code is from an allowed package.

        Similar to Django's app loading and FastAPI's dependency injection,
        this uses frame inspection to verify the caller's module.

        Parameters
        ----------
        allowed_packages : set[str]
            Set of package prefixes that are allowed to use the namespace.

        Returns
        -------
        bool
            True if caller is from an allowed package, False otherwise.
        """
        import sys

        # Get the call stack, skipping internal frames
        frame = sys._getframe(2)  # Skip this method and the calling method

        while frame is not None:
            module_name = frame.f_globals.get("__name__", "")

            # Skip test modules (similar to Django's test runner)
            if "test" in module_name or "pytest" in module_name:
                next_frame = frame.f_back
                if next_frame is None:
                    break
                frame = next_frame
                continue

            # Check if module is from an allowed package
            for allowed in allowed_packages:
                if module_name.startswith(allowed):
                    return True

            # Move up the stack to check parent callers
            next_frame = frame.f_back
            if next_frame is None:
                break
            frame = next_frame

        return False

    def _clear_for_testing(self, namespace: str | None = None) -> None:
        """[TESTING ONLY] Clear components from registry.

        This method should only be used in test environments to reset state
        between tests. Production code should treat the registry as append-only.

        Parameters
        ----------
        namespace : str | None
            If provided, only clear that namespace.
            If None, clear everything except 'core'.

        Raises
        ------
        RuntimeError
            If called in production environment.
        """
        import os
        import warnings

        # Only allow in test environments
        if not any(
            [
                os.environ.get("PYTEST_CURRENT_TEST"),
                os.environ.get("TEST_MODE"),
                "test" in os.environ.get("ENV", "").lower(),
            ]
        ):
            raise RuntimeError(
                "clear() is only available in test environments. "
                "Set PYTEST_CURRENT_TEST or TEST_MODE environment variable to use."
            )

        warnings.warn(
            "Clearing registry components. This should only be done in tests.",
            UserWarning,
            stacklevel=2,
        )

        with self._lock:
            if namespace:
                if namespace not in self._namespaces:
                    raise ValueError(f"Namespace '{namespace}' not registered")

                for comp_type in self._components:
                    if namespace in self._components[comp_type]:
                        self._components[comp_type][namespace].clear()

                logger.warning(f"[TEST] Cleared namespace '{namespace}'")
            else:
                # Clear all except core
                for comp_type in self._components:
                    namespaces = list(self._components[comp_type].keys())
                    for ns in namespaces:
                        if ns != "core":
                            del self._components[comp_type][ns]

                # Reset namespaces
                self._namespaces = {"core"}
                self._plugins_loaded = False

                logger.warning("[TEST] Cleared all non-core components")

    def get_dependents(self, component_name: str) -> set[str]:
        """Get all components that depend on the given component.

        Parameters
        ----------
        component_name : str
            Name of component (can include namespace).

        Returns
        -------
        set[str]
            Set of component names that depend on this component.

        Examples
        --------
        >>> # Check who depends on a component before replacing
        >>> dependents = registry.get_dependents('core:database_adapter')
        >>> if dependents:
        ...     print(f"Warning: {len(dependents)} components depend on this")
        """
        with self._lock:
            # Ensure namespace prefix
            if ":" not in component_name:
                component_name = f"core:{component_name}"

            return self._dependents_graph.get(component_name, set()).copy()

    def resolve_dependencies(self, component_name: str) -> list[str]:
        """Resolve component dependencies in initialization order.

        Parameters
        ----------
        component_name : str
            Name of component (can include namespace).

        Returns
        -------
        list[str]
            Ordered list of component names to initialize.

        Raises
        ------
        ValueError
            If circular dependencies detected.
        """
        visited = set()
        result = []
        temp_visited = set()

        def visit(name: str) -> None:
            if name in temp_visited:
                raise ValueError(f"Circular dependency detected involving '{name}'")
            if name in visited:
                return

            temp_visited.add(name)

            # Get dependencies
            deps = self._dependency_graph.get(name, set())
            for dep in deps:
                visit(dep)

            temp_visited.remove(name)
            visited.add(name)
            result.append(name)

        # Ensure namespace prefix
        if ":" not in component_name:
            component_name = f"core:{component_name}"

        visit(component_name)
        return result

    def validate_core_components(self) -> dict[str, list[str]]:
        """Validate that expected core components are registered.

        Returns
        -------
        dict[str, list[str]]
            Dictionary of component types to list of registered component names.

        Raises
        ------
        RuntimeError
            If expected core components are missing.
        """
        # Define minimum expected core components
        # These should always be available in a working hexDAG installation
        expected_core = {
            ComponentType.NODE: ["passthrough", "logging"],
            # Add more as core components are developed:
            # ComponentType.ADAPTER: ['http', 'file'],
            # ComponentType.TOOL: ['shell', 'python'],
        }

        registered_core: dict[str, list[str]] = {}
        missing = []

        for comp_type, expected_names in expected_core.items():
            registered = []
            for name in expected_names:
                try:
                    self.get(name, component_type=comp_type, namespace="core")
                    registered.append(name)
                except KeyError:
                    missing.append(f"{comp_type}:{name}")

            if registered:
                registered_core[comp_type.value] = registered

        if missing:
            logger.error(
                f"Missing expected core components: {missing}. "
                f"This may indicate an incomplete installation."
            )

        return registered_core

    @property
    def ready(self) -> bool:
        """Check if registry is ready for use."""
        return self._ready

    def _load_core_components(self) -> None:
        """Load core framework components and discover available plugins.

        Core components are always loaded eagerly and auto-register via decorators.
        Plugins are discovered but not loaded until needed (lazy loading).
        Only runs once on registry initialization.
        """
        # Define core modules that contain framework components
        # These are part of hexDAG itself, not user code or plugins
        core_modules = [
            "hexai.core.nodes",  # Core node components
            "hexai.core.adapters",  # Core adapters (if they exist)
            "hexai.core.tools",  # Core tools (if they exist)
            "hexai.core.policies",  # Core policies (if they exist)
            "hexai.core.memory",  # Core memory (if they exist)
            "hexai.core.observers",  # Core observers (if they exist)
        ]

        for module_name in core_modules:
            try:
                # Import module to trigger decorator registration
                __import__(module_name)
                logger.debug(f"Loaded core module: {module_name}")
            except ImportError:
                # It's OK if some core modules don't exist yet
                # Only log in debug mode to avoid noise
                logger.debug(f"Core module not found (OK if not implemented): {module_name}")
            except Exception as e:
                # But other errors should be logged as warnings
                logger.warning(f"Error loading core module {module_name}: {e}")

        # Discover available plugins (but don't load them yet - lazy loading)
        # This is the "manifest" for plugins - they declare themselves via entry points
        from .discovery import discover_entry_points

        self._available_plugins = discover_entry_points("hexai.plugins")
        if self._available_plugins:
            logger.info(f"Discovered {len(self._available_plugins)} plugins (lazy loading enabled)")

    def _register_pending_components(self) -> None:
        """Register any components that were decorated before registry init.

        This handles components decorated in modules imported before
        the registry singleton was created.
        """
        from hexai.core.registry.decorators import register_pending_components

        count = register_pending_components(self)
        if count > 0:
            logger.info(f"Registered {count} decorated components")

    def set_ready(self) -> None:
        """Mark registry as ready (called after app initialization)."""
        with self._lock:
            self._ready = True
            logger.info("Registry marked as ready")


# Create the singleton instance
registry = ComponentRegistry()

# Core components are automatically loaded when the registry is created.
# They are decorated with @node(namespace='core') or similar decorators
# in modules like hexai.core.nodes, hexai.core.adapters, etc.

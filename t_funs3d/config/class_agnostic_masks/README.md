Stage-one inference configs adapted from the vendored OpenMask3D
class_agnostic_mask_computation/conf directory (upstream MIT license).
Only the default single-scene configuration and its dependencies are copied.

Hydra package names and composition order are explicit. Working-directory
changes are disabled; the entry point already restores the original directory.
Mask3D receives its nested backbone config with `_recursive_: false` because
its constructor instantiates that backbone itself.

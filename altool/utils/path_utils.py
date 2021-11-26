#!/usr/bin/env python
"""
Description
===========
Helper functions are defined in this module.
"""
import os.path as osp


def module_path(module_name=''):
    """Get path of module.

    Args:
        module_name: Name of a module

    Example:
        >>> from mipack.utils import module_path
        >>> # get the root of this package.
        >>> module_path('.')
    """
    rootdir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    if module_name:
        return osp.join(rootdir, module_name)
    return rootdir

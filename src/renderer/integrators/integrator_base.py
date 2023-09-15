#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : integrator_base
# @Date : 2023-09-15-14-23
# @Project : nerf-torch
# @Author : seungmin

"""
Base class for integrators.
"""

import torch


class IntegratorBase(object):
    """
    Base class for integrators.
    """

    def __init__(self, *arg, **kwargs):
        pass

    def integrate_along_rays(
        self,
        sigma: torch.Tensor,
        radiance: torch.Tensor,
        delta: torch.Tensor,
    ):
        """
        Determines pixel colors given densities, interval length, and radiance values
        obtained along rays.
        """
        raise NotImplementedError()
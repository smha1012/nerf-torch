#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : signal_encoder_base
# @Date : 2023-09-15-14-02
# @Project : nerf-torch
# @Author : seungmin

import torch


class SignalEncoderBase:
    """
    Base class for signal encoders.
    """

    def __init__(self):
        pass

    def encode(self, in_signal: torch.Tensor) -> torch.Tensor:
        """
        Computes the encoding of the given signal.

        Args:
            in_signal: An instance of torch.Tensor of shape (N, C).
                Input signal being encoded.

        Returns:
            An instance of torch.Tensor of shape (N, self.out_dim).
                The encoding of the input signal.
        """
        raise NotImplementedError()
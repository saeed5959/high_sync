# TODO: Adapted from cli
from typing import Callable, List, Optional

import numpy as np


def ordered_halving(val):
    bin_str = f"{val:064b}"
    bin_flip = bin_str[::-1]
    as_int = int(bin_flip, 2)

    return as_int / (1 << 64)


def uniform(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    if num_frames <= context_size:
        yield list(range(num_frames))
        return

    context_stride = min(
        context_stride, int(np.ceil(np.log2(num_frames / context_size))) + 1
    )

    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(step)))
        for j in range(
            int(ordered_halving(step) * context_step) + pad,
            num_frames + pad + (0 if closed_loop else -context_overlap),
            (context_size * context_step - context_overlap),
        ):
            yield [
                e % num_frames
                for e in range(j, j + context_size * context_step, context_step)
            ]


def get_context_scheduler(name: str) -> Callable:
    if name == "uniform":
        return uniform
    else:
        raise ValueError(f"Unknown context_overlap policy {name}")


def get_total_steps(
    scheduler,
    timesteps: List[int],
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    return sum(
        len(
            list(
                scheduler(
                    i,
                    num_steps,
                    num_frames,
                    context_size,
                    context_stride,
                    context_overlap,
                )
            )
        )
        for i in range(len(timesteps))
    )

def modify_context_all(context):
    '''
    modify this context to remove first indexes in last context and expand it to 16 frames
    example : 
        input : [[240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 0, 1, 2, 3, 4, 5]]
        output : s = [[234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249]]
    '''
    context = context[0]

    if context[0]==0:
        len_context = len(context)
        context_new = [list(np.arange(0, 12))]

    else:
        len_context = 1
        max_value_list = context[0]
        for c, value in enumerate(context):
            if value > max_value_list:
                len_context = c + 1
                max_value_list = value

        context_new = [list(np.arange(max(0,max_value_list-12+1), max_value_list+1))]

    return context_new, len_context

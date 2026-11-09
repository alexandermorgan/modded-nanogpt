"""Which total step counts a shortened run can use, and the NUM_SCHEDULED_ITERATIONS for each.

    python -m track_1_short.step_counts 840    # the usable totals nearest 840
    python -m track_1_short.step_counts        # all of them

A run's total is NUM_SCHEDULED_ITERATIONS + SCHEDULE_GROWTH_STEPS + num_extension_iterations
(the record: 1122 + 52 + 20 = 1194). The stage boundaries scale with NUM_SCHEDULED_ITERATIONS, so a
shorter run moves them. A total is usable here when it keeps what the record keeps:
  - total ≡ 1194 (mod 4). The tail averages count their ticks back from the last step, every 4th
    step, so this keeps their phase against the period-4 n-gram and value_embeds updates.
  - every stage boundary on a step of the same parity as the record's (Adam params update on odd
    steps only; record #360 grew its run by 52 steps for the same reason).
  - the tail-average assert: total - VALUE_EMBED_AVG_WINDOW >= NGRAM_ADAM_PERIOD4_START.
Other totals run too; they just leave the record's tuned alignment.
"""
import sys

import torch

from track_1_short.config import (LR_COOLDOWN_FRAC, SCHEDULE_GROWTH_STEPS, SPLIT_EMBED_STAGE, TRAINING_STAGES,
                                  WS_POST_YARN_EXT, Hyperparameters)
from track_1_short.schedule import TrainingSchedule

# Copied from ngram_table.py and tail_average.py, whose imports pull in the GPU kernels; TailAverages
# asserts the same inequality against the real values at startup.
NGRAM_ADAM_PERIOD4_START = 336
VALUE_EMBED_AVG_WINDOW = 250

RECORD_SCHEDULED = 1122
EXTENSION = Hyperparameters().num_extension_iterations
FIXED = SCHEDULE_GROWTH_STEPS + EXTENSION  # steps a total has on top of NUM_SCHEDULED_ITERATIONS


def schedule(scheduled: int) -> TrainingSchedule:
    return TrainingSchedule(TRAINING_STAGES, scheduled, EXTENSION, torch.device("cpu"), LR_COOLDOWN_FRAC,
                            SPLIT_EMBED_STAGE, WS_POST_YARN_EXT)


def starts(s: TrainingSchedule) -> list[int]:
    return [start for start, _ in s.boundaries[1:]]


def usable_totals() -> list[int]:
    ref = schedule(RECORD_SCHEDULED)
    parity = [b % 2 for b in starts(ref)]
    out = []
    for scheduled in range(RECORD_SCHEDULED % 4, RECORD_SCHEDULED + 1, 4):
        total = scheduled + FIXED
        if total - VALUE_EMBED_AVG_WINDOW < NGRAM_ADAM_PERIOD4_START:
            continue
        if [b % 2 for b in starts(schedule(scheduled))] == parity:
            out.append(total)
    return out


def main() -> None:
    totals = usable_totals()
    if len(sys.argv) > 1:
        want = int(sys.argv[1])
        below = [t for t in totals if t <= want][-1:]
        above = [t for t in totals if t >= want][:1]
        totals = sorted(set(below + above))
    for total in totals:
        s = schedule(total - FIXED)
        print(f"total {total:5d}   NUM_SCHEDULED_ITERATIONS={total - FIXED:<5d} stage starts {starts(s)}")


if __name__ == "__main__":
    main()

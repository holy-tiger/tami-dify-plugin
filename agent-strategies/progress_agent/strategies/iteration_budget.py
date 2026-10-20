"""Validated runtime iteration budgets and execution state."""

import math
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class IterationBudget(BaseModel):
    maximum_iterations: int = Field(default=30, ge=1, le=500)
    progress_check_start: int = Field(default=20, ge=1, le=500)
    progress_check_interval: int = Field(default=5, ge=1, le=500)
    final_response_rounds: int = Field(default=1, ge=1, le=500)

    @field_validator(
        "maximum_iterations",
        "progress_check_start",
        "progress_check_interval",
        "final_response_rounds",
        mode="before",
    )
    @classmethod
    def parse_integer(cls, value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError("must be an integer, not a boolean")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and math.isfinite(value) and value.is_integer():
            return int(value)
        if isinstance(value, str) and re.fullmatch(r"[+-]?[0-9]+", value.strip()):
            return int(value.strip())
        raise ValueError(
            "must be an integer; resolve Chatflow variables before invoking the strategy"
        )

    @model_validator(mode="after")
    def validate_relationships(self) -> "IterationBudget":
        if self.final_response_rounds > self.maximum_iterations:
            raise ValueError("final_response_rounds must not exceed maximum_iterations")
        if "progress_check_start" not in self.model_fields_set:
            self.progress_check_start = min(
                self.progress_check_start,
                max(1, self.maximum_iterations - self.final_response_rounds),
            )
        if self.progress_check_start > self.maximum_iterations:
            raise ValueError("progress_check_start must not exceed maximum_iterations")
        return self

    def state(self, current_iteration: int) -> dict[str, Any]:
        if not 1 <= current_iteration <= self.maximum_iterations:
            raise ValueError("current iteration is outside the configured budget")
        finalization_start = self.maximum_iterations - self.final_response_rounds + 1
        is_finalization = current_iteration >= finalization_start
        is_checkpoint = (
            not is_finalization
            and current_iteration >= self.progress_check_start
            and (current_iteration - self.progress_check_start) % self.progress_check_interval == 0
        )
        return {
            "current_iteration": current_iteration,
            "maximum_iterations": self.maximum_iterations,
            "remaining_iterations_after_current": self.maximum_iterations - current_iteration,
            "remaining_tool_rounds_including_current": max(
                0, finalization_start - current_iteration
            ),
            "final_response_rounds": self.final_response_rounds,
            "is_checkpoint": is_checkpoint,
            "is_finalization": is_finalization,
        }

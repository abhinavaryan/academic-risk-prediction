"""Request / response models (also documented in the OpenAPI schema at /api/docs)."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
)

from riskml.modeling.zoo import SERVING_MODELS

FeatureValue = StrictFloat | StrictInt | StrictStr | None
ShortStr = Annotated[str, Field(min_length=1, max_length=200)]
ModelName = Literal[SERVING_MODELS]  # type: ignore[valid-type]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cutoff: Annotated[int, Field(strict=True, ge=0, le=400)]
    features: Annotated[dict[ShortStr, FeatureValue], Field(max_length=200)] = Field(
        default_factory=dict
    )


class FactorOut(BaseModel):
    name: str
    label: str
    value: float | str | None
    baseline: float | str | None
    impact: float
    direction: Literal["increases", "decreases"]


class PredictResponse(BaseModel):
    cutoff: int
    probability: float
    band: Literal["low", "moderate", "high"]
    at_risk: bool
    threshold: float
    factors: list[FactorOut]
    filled_with_baseline: list[str]
    model: str
    model_version: str


class TrainConfigIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: ShortStr
    positive_values: Annotated[list[ShortStr], Field(max_length=100)] = Field(default_factory=list)
    negative_values: Annotated[list[ShortStr], Field(max_length=100)] | None = None
    unknown_values: Annotated[list[ShortStr], Field(max_length=100)] = Field(default_factory=list)
    positive_below: float | None = None
    id_column: ShortStr | None = None
    exclude: Annotated[list[ShortStr], Field(max_length=200)] = Field(default_factory=list)
    models: (
        Annotated[list[ModelName], Field(min_length=1, max_length=len(SERVING_MODELS))] | None
    ) = None

    @field_validator("positive_below")
    @classmethod
    def _finite(cls, v: float | None) -> float | None:
        if v is not None and not math.isfinite(v):
            raise ValueError("must be a finite number")
        return v

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class CreateRazorpayOrderRequest(BaseModel):
    """Create a Razorpay order (USD). User is always taken from auth token."""

    credit_type: Literal["plan", "credit"]
    plan_type: Optional[Literal["silver", "gold", "platinum"]] = None
    timeperiod: Optional[Literal["monthly", "yearly"]] = None
    credit: float = Field(..., gt=0, description="Credits to add on successful payment")
    amount: float = Field(
        ...,
        gt=0,
        description=(
            "Total amount in US dollars. Backend converts USD -> RAZORPAY_CURRENCY "
            "in real-time and sends converted minor amount to Razorpay."
        ),
    )

    @model_validator(mode="after")
    def _plan_fields(self) -> "CreateRazorpayOrderRequest":
        if self.credit_type == "plan":
            if self.plan_type is None or self.timeperiod is None:
                raise ValueError("plan_type and timeperiod are required when credit_type is 'plan'")
        else:
            if self.plan_type is not None or self.timeperiod is not None:
                # allow null; ignore if client sends them — clear for consistency
                pass
        return self


class VerifyRazorpayPaymentRequest(BaseModel):
    stylic_payment_id: str = Field(..., min_length=1)
    razorpay_order_id: str = Field(..., min_length=1)
    razorpay_payment_id: str = Field(..., min_length=1)
    razorpay_signature: str = Field(..., min_length=1)

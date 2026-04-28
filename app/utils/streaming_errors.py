"""Shared user-facing messages for streaming generation failures."""

CUSTOM_INPUT_POLICY_ERROR_MESSAGE = (
    "Your input image was violating our company policy, so please try with a different image."
)


def custom_input_policy_error_payload() -> dict[str, str]:
    return {
        "step": "error",
        "message": CUSTOM_INPUT_POLICY_ERROR_MESSAGE,
    }

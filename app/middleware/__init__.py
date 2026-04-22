"""HTTP middleware for production-grade security & observability.

Everything in this package is backend-only: adding these middlewares does
NOT change request or response *bodies*. It only:

- Adds/removes HTTP headers (security headers, server banner, request ID).
- Rejects requests that are obviously abusive (oversize body, untrusted
  Host, bad admin key on ops endpoints).

So existing frontend clients keep working unchanged.
"""

from app.middleware.security_headers import SecurityHeadersMiddleware
from app.middleware.request_context import RequestContextMiddleware
from app.middleware.body_size_limit import BodySizeLimitMiddleware
from app.middleware.admin_guard import AdminKeyGuardMiddleware

__all__ = [
    "SecurityHeadersMiddleware",
    "RequestContextMiddleware",
    "BodySizeLimitMiddleware",
    "AdminKeyGuardMiddleware",
]

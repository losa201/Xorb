from fastapi import FastAPI
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.gzip import GZipMiddleware
from fastapi.middleware import Middleware
from fastapi.middleware import Middleware
from fastapi.middleware import Middleware

def configure_security_headers(app: FastAPI):
    """Configure strict security headers for the application."""
    
    # Content Security Policy with strict rules
    csp_policy = {
        'default-src': "'self'",
        'script-src': "'self' 'unsafe-eval' https://trusted-cdn.com",
        'style-src': "'self' 'unsafe-inline' https://trusted-cdn.com",
        'img-src': "'self' data: https://trusted-images.com",
        'font-src': "'self' https://trusted-fonts.com",
        'connect-src': "'self' https://api.trusted-service.com",
        'object-src': "'none'",
        'base-uri': "'self'",
        'form-action': "'self'",
        'frame-ancestors': "'none'",
        'worker-src': "'self' blob:"
    }

    # Convert CSP policy to string
    csp_header = "; ".join([f"{key} {value}" for key, value in csp_policy.items()])

    # Add security headers middleware
    app.add_middleware(
        Middleware(
            "starlette.middleware.TrustedHostMiddleware",
            allowed_hosts=["xorb.security", "api.xorb.security", "localhost"]
        )
    )

    # Add CORS middleware with strict policies
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://xorb.security", "https://api.xorb.security", "https://localhost:3000"],
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
        expose_headers=["X-Content-Type-Options", "X-Frame-Options"],
        allow_credentials=True,
        max_age=600
    )

    # Add HSTS middleware
    app.add_middleware(
        Middleware(
            "starlette.middleware.http.HttpsRedirectMiddleware" if app.debug else "starlette.middleware.http.HSTSMiddleware",
            hsts_max_age=31536000,  # 1 year
            hsts_include_subdomains=True,
            hsts_preload=True
        )
    )

    # Add security headers to all responses
    @app.middleware("http")
    async def add_security_headers(request, call_next):
        response = await call_next(request)
        
        # Add Content Security Policy
        response.headers["Content-Security-Policy"] = csp_header
        
        # Add other security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "accelerometer=(), ambient-light-sensor=(), autoplay=(), battery=(), camera=(), "
            "display-capture=(), document-domain=(), encrypted-media=(), execution-while-not-rendered=(), "
            "execution-while-offscreen=(), fullscreen=(), geolocation=(), gyroscope=(), magnetometer=(), "
            "microphone=(), midi=(), navigation-override=(), payment=(), picture-in-picture=(), "
            "publickey-credentials-get=(), screen-wake-lock=(), sync-xhr=(), usb=(), web-share=(), "
            "xr-spatial-tracking=()"
        )
        
        # Remove server information
        response.headers["Server"] = "Xorb-Security-Platform"
        
        return response

    return app
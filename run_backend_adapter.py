from __future__ import annotations

import os

from gcp_agent_runtime.adapter import create_fastapi_app


def main() -> None:
    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "uvicorn is required to run backend adapter. Install requirements-gcp.txt."
        ) from exc

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    app = create_fastapi_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

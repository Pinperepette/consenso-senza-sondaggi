"""Flask app factory."""
from __future__ import annotations

from flask import Flask

from consenso.api.routes import api


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates",
                static_folder="static", static_url_path="/static")
    app.register_blueprint(api)
    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    app.run(host="0.0.0.0", port=5000, debug=True)

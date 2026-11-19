"""Configure the managed workspace and verify observable YiGraph behavior."""

import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("demo", "configure", "neo4j", "engine", "apply-config", "serve"))
    parser.add_argument("--neo-home", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.action == "demo":
        from yigraph.demo import run_demo

        result = run_demo()
        if result["shortest_path_A_to_D"] != ["A", "C", "D"]:
            raise RuntimeError(f"Unexpected shortest path: {result}")
        ranks = result["pagerank"]
        if set(ranks) != {"A", "B", "C", "D"} or abs(sum(ranks.values()) - 1) > 1e-6:
            raise RuntimeError(f"Unexpected PageRank result: {ranks}")
        print(json.dumps(result, indent=2))
    elif args.action == "configure":
        if args.neo_home is None:
            parser.error("configure requires --neo-home")
        configure(args.neo_home)
    elif args.action == "neo4j":
        verify_neo4j()
    elif args.action == "engine":
        from yigraph import AAGEngine

        print("Engine import OK")
    elif args.action == "apply-config":
        if args.config is None or args.source is None:
            parser.error("apply-config requires --config and --source")
        apply_config(args.config, args.source)
    else:
        if args.source is None:
            parser.error("serve requires --source")
        serve_web(args.source)


def apply_config(settings_path, source):
    import yaml
    from datetime import datetime, timezone

    settings = json.loads(settings_path.read_text(encoding="utf-8-sig"))
    config_path = Path(os.environ["YIGRAPH_HOME"]) / "config/engine_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    llm = config["reasoner"]["llm"]
    llm["provider"] = "openai"
    llm["openai"].update(
        base_url=settings["api_base_url"].rstrip("/") + "/",
        api_key=settings["api_key"], model=settings["model_name"],
    )
    config["retrieval"]["embedding"]["device"] = "cpu"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    models_path = source / "web/frontend/models.json"
    models = json.loads(models_path.read_text(encoding="utf-8")) if models_path.exists() else []
    model = next((item for item in models if item["name"] == settings["model_name"]), None)
    if model is None:
        model = {"id": max((item["id"] for item in models), default=0) + 1, "created_at": datetime.now(timezone.utc).isoformat()}
        models.append(model)
    model.update(name=settings["model_name"], base_url=settings["api_base_url"].rstrip("/"), api_key=settings["api_key"])
    models_path.write_text(json.dumps(models, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Model settings applied to the engine and Web interface.")


def serve_web(source):
    from werkzeug.serving import make_server

    os.chdir(source)
    sys.path.insert(0, str(source / "web/frontend"))
    from route import create_app

    app = create_app()
    server = make_server("127.0.0.1", 0, app, threaded=True)
    state = Path(os.environ["YIGRAPH_HOME"]).parent / "web-server.json"
    pending = state.with_suffix(".tmp")
    pending.write_text(json.dumps({"pid": os.getpid(), "url": f"http://127.0.0.1:{server.server_port}"}), encoding="utf-8")
    pending.replace(state)
    print(f"YiGraph is ready at http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def configure(neo_home):
    import yaml

    config_path = Path(os.environ["YIGRAPH_HOME"]) / "config/engine_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    neo = config["retrieval"]["database"]["neo4j"]
    if neo.get("uri") not in {"bolt://localhost:7687", "bolt://127.0.0.1:7687"}:
        raise RuntimeError("Workspace uses an external Neo4j URI. Use a separate install root for the managed local database; existing configuration was preserved.")
    # Preserve credentials on retries; never reset an initialized database.
    if not neo.get("password"):
        neo.update(enabled=True, uri="bolt://localhost:7687", user="neo4j", password=secrets.token_hex(20))
    neo["enabled"] = True
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    if not (neo_home / "data/dbms/auth").exists():
        result = subprocess.run(
            [str(neo_home / "bin/neo4j-admin.bat"), "set-initial-password", neo["password"]],
            capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError("Neo4j initial password setup failed; check JDK 11 and the database directory.")
    conf_path = neo_home / "conf/neo4j.conf"
    lines = conf_path.read_text(encoding="utf-8-sig").splitlines()
    settings = {
        "dbms.connectors.default_listen_address": "127.0.0.1",
        "dbms.connectors.default_advertised_address": "localhost",
        "dbms.connector.bolt.listen_address": "127.0.0.1:7687",
        "dbms.connector.http.listen_address": "127.0.0.1:7474",
        "dbms.connector.https.enabled": "false",
    }
    kept = [line for line in lines if line.split("=", 1)[0].strip() not in settings]
    conf_path.write_text("\n".join(kept + [f"{key}={value}" for key, value in settings.items()]) + "\n", encoding="utf-8")
    print("Local Neo4j configured; credentials stored only in the workspace configuration.")


def verify_neo4j():
    import yaml
    from neo4j import GraphDatabase

    config_path = Path(os.environ["YIGRAPH_HOME"]) / "config/engine_config.yaml"
    neo = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))["retrieval"]["database"]["neo4j"]
    deadline = time.monotonic() + 90
    last_error = None
    while time.monotonic() < deadline:
        try:
            with GraphDatabase.driver(neo["uri"], auth=(neo["user"], neo["password"]), connection_timeout=3) as driver:
                with driver.session() as session:
                    value = session.run("RETURN 1 AS value").single()["value"]
                    if value != 1:
                        raise RuntimeError("Unexpected database query result")
            print("Neo4j authenticated query OK")
            return
        except Exception as exc:
            last_error = type(exc).__name__
            if last_error in {"AuthError", "AuthenticationError"}:
                break
            time.sleep(2)
    raise RuntimeError(f"Neo4j verification failed ({last_error}); check logs/neo4j.*.log and local credentials.")


if __name__ == "__main__":
    main()

from dataclasses import dataclass, asdict
import json
from typing import Dict, Any
import yaml

@dataclass
class LLMConfig:
    provider: str
    ollama: Dict[str, Any]
    openai: Dict[str, Any]

@dataclass
class ReasonerConfig:
    llm: LLMConfig

@dataclass
class DatabaseConfig:
    graph: Dict[str, Any]
    vector: Dict[str, Any]
    neo4j: Dict[str, Any]  # Neo4j 图查询配置

@dataclass
class RagConfig:
    graph: Dict[str, Any]
    vector: Dict[str, Any]

@dataclass
class RetrievalConfig:
    database: DatabaseConfig
    embedding: Dict[str, Any]
    rag: RagConfig

@dataclass
class EngineConfig:
    version: str
    mode: str
    reasoner: ReasonerConfig
    retrieval: RetrievalConfig
    monitoring: Dict[str, Any]
    output: Dict[str, Any]


def load_config_from_yaml(config_path: str) -> EngineConfig:
    """从YAML配置文件加载配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        version = str(config_data.get("version", "1.0"))

        mode = str(config_data.get("mode", "interactive"))

        # reasoner.llm
        llm_section = (config_data or {}).get("reasoner", {}).get("llm", {})
        llm_config = LLMConfig(
            provider=llm_section.get("provider", "ollama"),
            ollama=llm_section.get("ollama", {}) or {},
            openai=llm_section.get("openai", {}) or {}
        )
        reasoner_config = ReasonerConfig(llm=llm_config)

        # retrieval
        retrieval_section = (config_data or {}).get("retrieval", {})
        database_section = retrieval_section.get("database", {})
        rag_section = retrieval_section.get("rag", {})
        #add gjq
        database_config = DatabaseConfig(
            graph=database_section.get("graph", {}) or {},
            vector=database_section.get("vector", {}) or {},
            neo4j=database_section.get("neo4j", ) or {}
        )
        rag_config = RagConfig(
            graph=rag_section.get("graph", {}) or {},
            vector=rag_section.get("vector", {}) or {}
        )
        retrieval_config = RetrievalConfig(
            database=database_config,
            embedding=retrieval_section.get("embedding", {}) or {},
            rag=rag_config
        )

        monitoring_config = config_data.get("monitoring", {}) or {}
        output_config = config_data.get("output", {}) or {}

        return EngineConfig(
            version=version,
            mode=mode,
            reasoner=reasoner_config,
            retrieval=retrieval_config,
            monitoring=monitoring_config,
            output=output_config
        )
    except Exception as e:
        raise ValueError(f"Failed to load config from {config_path}: {e}")


# 命令行配置函数
# TODO(chaoyi): 参数过多， llm 现在不支持通过命令行适配 可选的llm，后期需要修正 
def create_engine_config(
    graph_space_name: str = "graphllm_space",
    vector_collection_name: str = "graphllm_collection",
    llm_model: str = "llama3.1:70b",
    embedding_model: str = "BAAI/bge-large-en-v1.5",
    **kwargs
) -> EngineConfig:
    """创建Engine配置"""
    
    # reasoner.llm
    provider = kwargs.get("llm_provider", "ollama")
    if provider == "openai":
        llm = LLMConfig(
            provider="openai",
            ollama={
                "model_name": kwargs.get("ollama_model_name", llm_model),
                "device": kwargs.get("ollama_device", "cuda:0"),
                "timeout": kwargs.get("ollama_timeout", 150000),
                "port": kwargs.get("ollama_port", 11434),
            },
            openai={
                "base_url": kwargs.get("openai_base_url", "https://api.openai.com/v1"),
                "api_key": kwargs.get("openai_api_key"),
                "model": kwargs.get("openai_model", "gpt-4o"),
            },
        )
    else:
        llm = LLMConfig(
            provider="ollama",
            ollama={
                "model_name": kwargs.get("ollama_model_name", llm_model),
                "device": kwargs.get("ollama_device", "cuda:0"),
                "timeout": kwargs.get("ollama_timeout", 150000),
                "port": kwargs.get("ollama_port", 11434),
            },
            openai={
                "base_url": kwargs.get("openai_base_url", "https://api.openai.com/v1"),
                "api_key": kwargs.get("openai_api_key"),
                "model": kwargs.get("openai_model", "gpt-4o"),
            },
        )

    reasoner = ReasonerConfig(llm=llm)

    database = DatabaseConfig(
        graph={
            "space_name": graph_space_name,
            "server_ip": kwargs.get("graph_server_ip", "127.0.0.1"),
            "server_port": kwargs.get("graph_server_port", "9669"),
            "create": kwargs.get("create_graph_space", True),
            "verbose": kwargs.get("graph_verbose", False),
        },
        vector={
            "collection_name": vector_collection_name,
            "dim": kwargs.get("vector_dim", 1024),
            "host": kwargs.get("vector_host", "localhost"),
            "port": kwargs.get("vector_port", 19530),
        },
        neo4j=kwargs.get("neo4j", {}) or {},
    )

    rag = RagConfig(
        graph={
            "k_hop": kwargs.get("graph_k_hop", 2),
            "pruning": kwargs.get("graph_pruning", 30),
            "data_type": kwargs.get("graph_data_type", "qa"),
            "pruning_mode": kwargs.get("graph_pruning_mode", "embedding_for_perentity"),
        },
        vector={
            "k_similarity": kwargs.get("vector_k_similarity", 5),
        },
    )

    retrieval = RetrievalConfig(
        database=database,
        embedding={
            "model_name": embedding_model,
            "batch_size": kwargs.get("embed_batch_size", 20),
            "device": kwargs.get("embed_device", "cuda:0"),
        },
        rag=rag,
    )

    monitoring = {
        "enable_metrics": kwargs.get("enable_metrics", True),
        "log_level": kwargs.get("log_level", "INFO"),
        "save_metrics": kwargs.get("save_metrics", True),
        "metrics_file": kwargs.get("metrics_file", "engine_metrics.json"),
    }

    output = {
        "save_intermediate_results": kwargs.get("save_intermediate_results", True),
        "output_dir": kwargs.get("output_dir", "./output"),
        "log_file": kwargs.get("log_file", "./engine.log"),
    }

    return EngineConfig(
        version=str(kwargs.get("version", "1.0")),
        reasoner=reasoner,
        retrieval=retrieval,
        monitoring=monitoring,
        output=output,
    )


if __name__ == "__main__":
    def _summarize(cfg: EngineConfig) -> str:
        try:
            d = asdict(cfg)
            provider = d.get("reasoner", {}).get("llm", {}).get("provider")
            ollama_model = d.get("reasoner", {}).get("llm", {}).get("ollama", {}).get("model_name")
            openai_model = d.get("reasoner", {}).get("llm", {}).get("openai", {}).get("model")
            graph_space = d.get("retrieval", {}).get("database", {}).get("graph", {}).get("space_name")
            vector_collection = d.get("retrieval", {}).get("database", {}).get("vector", {}).get("collection_name")
            embed_model = d.get("retrieval", {}).get("embedding", {}).get("model_name")
            return (
                f"version={d.get('version')} | provider={provider} | "
                f"ollama_model={ollama_model} | openai_model={openai_model} | "
                f"graph_space={graph_space} | vector_collection={vector_collection} | "
                f"embedding={embed_model}"
            )
        except Exception as e:
            return f"<summary error: {e}>"

    def test_config_main():
        yaml_path = "/home/chency/GraphLLM/config/engine_config.yaml"
        print("[Test] load_config_from_yaml ->", yaml_path)
        cfg_from_yaml = load_config_from_yaml(yaml_path)
        print("[OK] ", _summarize(cfg_from_yaml))
        print("[FULL YAML CONFIG]\n" + json.dumps(asdict(cfg_from_yaml), indent=2, ensure_ascii=False))

        print("[Test] create_engine_config (defaults)")
        cfg_default = create_engine_config()
        print("[OK] ", _summarize(cfg_default))
        print("[FULL DEFAULT CONFIG]\n" + json.dumps(asdict(cfg_default), indent=2, ensure_ascii=False))

    test_config_main()
import networkx as nx
import inspect

sig = inspect.signature(nx.pagerank)
print("Pagerank 函数的所有参数：")
for param_name, param in sig.parameters.items():
    print(f"  - {param_name}: {param}")

print("\nDocstring:")
print(inspect.getdoc(nx.pagerank))

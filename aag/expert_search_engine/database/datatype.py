from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class VertexData:
    vid: str
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to standard dictionary structure for JSON transmission or algorithm input"""
        return {
            "vid": self.vid,
            "properties": self.properties
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "VertexData":
        """Deserialize from dictionary to VertexData"""
        return VertexData(
            vid=data.get("vid"),
            properties=data.get("properties", {})
        )

    def __repr__(self):
        return f"VertexData(vid={self.vid}, props={list(self.properties.keys())})"


@dataclass
class EdgeData:
    src: str
    dst: str
    rank: Optional[str] = None  # Optional rank value
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to standard dictionary structure for JSON transmission or algorithm input"""
        return {
            "src": self.src,
            "dst": self.dst,
            "rank": self.rank,
            "properties": self.properties
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "EdgeData":
        """Deserialize from dictionary to EdgeData"""
        return EdgeData(
            src=data.get("src"),
            dst=data.get("dst"),
            rank=data.get("rank"),
            properties=data.get("properties", {})
        )

    def __repr__(self):
        return f"EdgeData({self.src}->{self.dst}, props={list(self.properties.keys())})"


@dataclass    
class GraphData:
    def __init__(self, vertices: List[VertexData], edges: List[EdgeData]):
        self.vertices = vertices
        self.edges = edges

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to standard dictionary structure for JSON transmission or algorithm input"""
        return {
            "vertices": [v.to_dict() for v in self.vertices],
            "edges": [e.to_dict() for e in self.edges]
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "GraphData":
        """Deserialize from dictionary to GraphData"""
        vertices = [VertexData.from_dict(v) for v in data.get("vertices", [])]
        edges = [EdgeData.from_dict(e) for e in data.get("edges", [])]
        return GraphData(vertices=vertices, edges=edges)

    def __repr__(self):
        return f"GraphData(num_vertices={len(self.vertices)}, num_edges={len(self.edges)})"
    
    def has_vertex(self, vid: str) -> bool:
        for vertex in self.vertices:
            if vertex.vid == vid:
                return True
        return False
    
    def get_vertex_property(self, vid: str, property_name: str) -> Optional[Any]:
        """
        Get the property value of the specified property field for the specified vertex ID
        
        Args:
            vid: Vertex ID
            property_name: Property name
            
        Returns:
            Property value, returns None if vertex or property does not exist
        """
        for vertex in self.vertices:
            if vertex.vid == vid:
                if property_name == 'properties':
                    logger.info(f"Returning all properties for vertex {vid} {vertex.properties}")
                    return vertex.properties
                else:
                    # Otherwise return specific property
                    logger.info(f"Returning a properties for vertex {vid} {vertex.properties.get(property_name)}")
                    return vertex.properties.get(property_name)
        return None
    
    def get_edge_property(self, src: str, dst: str, property_name: str, rank: Optional[int] = None) -> Optional[Any]:
        """
        Get the property value of the specified property field for the specified edge
        
        Edges are distinguished by src, dst, and rank. If rank is None, only src and dst are used to distinguish.
        
        Args:
            src: Source vertex ID
            dst: Destination vertex ID
            property_name: Property name
            rank: Edge rank value, optional
            
        Returns:
            Property value, returns None if edge or property does not exist
        """
        for edge in self.edges:
            if edge.src == src and edge.dst == dst:
                # If rank is specified, rank must also match
                if rank is not None:
                    if edge.rank == rank:
                        if property_name == 'properties':
                            logger.info(f"Returning all properties for edge {edge} {edge.properties}")
                            return edge.properties
                        else:
                            # Otherwise return specific property
                            logger.info(f"Returning a properties for edge {edge} {edge.properties.get(property_name)}")
                            return edge.properties.get(property_name)
                else:
                    # If rank is not specified, only match src and dst
                    if property_name == 'properties':
                        logger.info(f"Returning all properties for edge {edge} {edge.properties}")
                        return edge.properties
                    else:
                        # Otherwise return specific property
                        logger.info(f"Returning a properties for edge {edge} {edge.properties.get(property_name)}")
                        return edge.properties.get(property_name)
        return None

    def get_edges_by_vertices(self, vertex_ids: set[str]) -> List[EdgeData]:
        """
        Get related edges based on vertex set (OR condition)
        
        Returns all edges where source OR target vertices are in the given vertex set.
        This includes edges that connect vertices within the set to vertices outside the set.
        
        Example:
            If vertex_ids = {A, B, C} and edges are A->B, B->C, C->D:
            Returns: [A->B, B->C, C->D] (C->D is included because C is in the set)
        
        Args:
            vertex_ids: Set of vertex IDs
            
        Returns:
            List of related edges (includes edges connecting to vertices outside the set)
        """
        related_edges = []
        for edge in self.edges:
            if edge.src in vertex_ids or edge.dst in vertex_ids:
                related_edges.append(edge)
        return related_edges
        
    def get_src_dst_by_vertices(self, vertex_ids: set[str]) -> List[EdgeData]:
        """
        Get related edges based on vertex set (AND condition)
        
        Returns all edges where BOTH source AND target vertices are in the given vertex set.
        This only includes edges that are completely within the vertex set (subgraph edges).
        
        Example:
            If vertex_ids = {A, B, C} and edges are A->B, B->C, C->D:
            Returns: [A->B, B->C] (C->D is excluded because D is not in the set)
        
        Args:
            vertex_ids: Set of vertex IDs
            
        Returns:
            List of related edges (only edges within the set, subgraph edges)
        """
        related_edges = []
        for edge in self.edges:
            if edge.src in vertex_ids and edge.dst in vertex_ids:
                related_edges.append(edge)
        return related_edges

    def get_vertices_by_edges(self, edges: List[EdgeData]) -> List[VertexData]:
        """
        Get related vertices based on edge set
        
        Returns all vertices that serve as source or target nodes in the given edges
        
        Args:
            edges: List of edges
            
        Returns:
            List of related vertices
        """
        vertex_ids = set()
        # Collect all related vertex IDs
        for edge in edges:
            vertex_ids.add(edge.src)
            vertex_ids.add(edge.dst)
        
        # Find corresponding vertex objects by vertex ID
        related_vertices = []
        for vertex in self.vertices:
            if vertex.vid in vertex_ids:
                related_vertices.append(vertex)
                
        return related_vertices

    def get_vertex_properties_schema(self) -> Dict[str, str]:
        """
        Get property field names and types for all vertices
        
        Args:
            graph_data: Graph data object
            
        Returns:
            Dictionary with property names as keys and property type names as values
        """
        schema = {}
        vertex=self.vertices[0]
        for prop_name, prop_value in vertex.properties.items():
                # Get property type
                prop_type = type(prop_value).__name__
                # If property name already exists, check if types are consistent
                if prop_name in schema:
                    if schema[prop_name] != prop_type:
                        logger.warning(f"Vertex property '{prop_name}' type inconsistent: {schema[prop_name]} vs {prop_type}")
                        # If inconsistent, mark as mixed type
                        schema[prop_name] = f"mixed({schema[prop_name]},{prop_type})"
                else:
                    schema[prop_name] = prop_type       
        logger.info(f"Vertex property schema: {schema}")
        return schema

    def get_edge_properties_schema(self) -> Dict[str, str]:
        """
        Get property field names and types for all edges
        
        Args:
            graph_data: Graph data object
            
        Returns:
            Dictionary with property names as keys and property type names as values
        """
        schema = {}
        edge=self.edges[0]
        for prop_name, prop_value in edge.properties.items():
                # Get property type
                prop_type = type(prop_value).__name__
                # If property name already exists, check if types are consistent
                if prop_name in schema:
                    if schema[prop_name] != prop_type:
                        logger.warning(f"Edge property '{prop_name}' type inconsistent: {schema[prop_name]} vs {prop_type}")
                        # If inconsistent, mark as mixed type
                        schema[prop_name] = f"mixed({schema[prop_name]},{prop_type})"
                else:
                    schema[prop_name] = prop_type              
        logger.info(f"Edge property schema: {schema}")
        return schema

    def get_graph_schema(self) -> Dict[str, Any]:
        """
        Get complete schema information of the graph (vertex and edge property schemas)
        
        Args:
            graph_data: Graph data object
            
        Returns:
            Dictionary containing vertex and edge schemas
        """
        return {
            "vertex_schema": self.get_vertex_properties_schema(),
            "edge_schema": self.get_edge_properties_schema()
        }


        
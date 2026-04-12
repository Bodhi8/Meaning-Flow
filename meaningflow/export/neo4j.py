"""
Neo4j export for MeaningFlow semantic graphs.

Writes clusters as nodes and inter-cluster relationships as edges
to a Neo4j graph database. Supports both the Neo4j Python driver
and Cypher script export for environments without a live connection.

Requirements:
    pip install neo4j

Usage:
    from meaningflow import SemanticGraph
    from meaningflow.export import to_neo4j, to_cypher

    sg = SemanticGraph(texts=my_texts, embedder="all-MiniLM-L6-v2")
    sg.fit()

    # Direct write to Neo4j
    to_neo4j(sg, uri="bolt://localhost:7687", auth=("neo4j", "password"))

    # Or export as a Cypher script file
    to_cypher(sg, path="meaningflow_graph.cypher")
"""

import logging
from typing import Optional, Tuple, List

log = logging.getLogger(__name__)


def to_neo4j(
    graph,
    uri: str = "bolt://localhost:7687",
    auth: Tuple[str, str] = ("neo4j", "neo4j"),
    database: str = "neo4j",
    clear_existing: bool = False,
    label_prefix: str = "MF",
) -> dict:
    """
    Write a fitted SemanticGraph to a Neo4j database.

    Each cluster becomes a node with label `{label_prefix}_Cluster`.
    Each inter-cluster edge becomes a `SIMILAR_TO` relationship with
    a similarity weight property.

    Parameters
    ----------
    graph : SemanticGraph
        A fitted SemanticGraph instance.
    uri : str
        Neo4j Bolt URI. Default: bolt://localhost:7687.
    auth : tuple of (str, str)
        Username and password.
    database : str
        Target database name. Default: 'neo4j'.
    clear_existing : bool
        If True, delete all nodes with the label prefix before writing.
    label_prefix : str
        Prefix for Neo4j node labels. Default: 'MF'.

    Returns
    -------
    dict
        Summary with counts of nodes and relationships created.
    """
    try:
        from neo4j import GraphDatabase
    except ImportError:
        raise ImportError(
            "Neo4j driver not installed. Run: pip install neo4j"
        )

    if not graph._is_fitted:
        raise RuntimeError("SemanticGraph must be fitted before export. Call .fit() first.")

    driver = GraphDatabase.driver(uri, auth=auth)
    node_label = f"{label_prefix}_Cluster"
    nodes_created = 0
    rels_created = 0

    with driver.session(database=database) as session:
        # Optionally clear existing data with this prefix
        if clear_existing:
            session.run(f"MATCH (n:{node_label}) DETACH DELETE n")
            log.info("Cleared existing %s nodes", node_label)

        # Create cluster nodes
        for cluster in graph.clusters:
            props = {
                "cluster_id": cluster.id,
                "size": cluster.size,
                "volume": cluster.volume,
                "top_terms": cluster.top_terms[:10],
                "label": cluster.top_terms[0] if cluster.top_terms else f"Cluster {cluster.id}",
            }
            session.run(
                f"CREATE (n:{node_label} $props)",
                props=props,
            )
            nodes_created += 1

        # Create an index for fast lookups
        session.run(
            f"CREATE INDEX IF NOT EXISTS FOR (n:{node_label}) ON (n.cluster_id)"
        )

        # Create relationships from the NetworkX graph
        if graph.graph is not None:
            for u, v, data in graph.graph.edges(data=True):
                weight = data.get("weight", 0.0)
                session.run(
                    f"""
                    MATCH (a:{node_label} {{cluster_id: $u}})
                    MATCH (b:{node_label} {{cluster_id: $v}})
                    CREATE (a)-[:SIMILAR_TO {{weight: $weight}}]->(b)
                    """,
                    u=u, v=v, weight=weight,
                )
                rels_created += 1

    driver.close()

    summary = {
        "nodes_created": nodes_created,
        "relationships_created": rels_created,
        "node_label": node_label,
        "database": database,
    }
    log.info(
        "Neo4j export complete: %d nodes, %d relationships",
        nodes_created, rels_created,
    )
    return summary


def to_cypher(
    graph,
    path: str = "meaningflow_graph.cypher",
    label_prefix: str = "MF",
) -> str:
    """
    Export a fitted SemanticGraph as a Cypher script file.

    Useful when you don't have a live Neo4j connection but want to
    generate the import script for later use.

    Parameters
    ----------
    graph : SemanticGraph
        A fitted SemanticGraph instance.
    path : str
        Output file path. Default: 'meaningflow_graph.cypher'.
    label_prefix : str
        Prefix for Neo4j node labels. Default: 'MF'.

    Returns
    -------
    str
        The generated Cypher script as a string.
    """
    if not graph._is_fitted:
        raise RuntimeError("SemanticGraph must be fitted before export. Call .fit() first.")

    node_label = f"{label_prefix}_Cluster"
    lines = []

    # Header
    lines.append(f"// MeaningFlow Semantic Graph Export")
    lines.append(f"// Clusters: {len(graph.clusters)}")
    lines.append(f"// Edges: {graph.graph.number_of_edges() if graph.graph else 0}")
    lines.append("")

    # Clear existing
    lines.append(f"// Clear existing MeaningFlow data")
    lines.append(f"MATCH (n:{node_label}) DETACH DELETE n;")
    lines.append("")

    # Create nodes
    lines.append(f"// Create cluster nodes")
    for cluster in graph.clusters:
        terms_escaped = [t.replace("'", "\\'").replace('"', '\\"') for t in cluster.top_terms[:10]]
        terms_str = "[" + ", ".join(f'"{t}"' for t in terms_escaped) + "]"
        label_escaped = (cluster.top_terms[0] if cluster.top_terms else f"Cluster {cluster.id}").replace('"', '\\"')

        lines.append(
            f'CREATE (:{node_label} {{'
            f'cluster_id: {cluster.id}, '
            f'size: {cluster.size}, '
            f'volume: {cluster.volume}, '
            f'label: "{label_escaped}", '
            f'top_terms: {terms_str}'
            f'}});'
        )
    lines.append("")

    # Create index
    lines.append(f"// Create index")
    lines.append(f"CREATE INDEX IF NOT EXISTS FOR (n:{node_label}) ON (n.cluster_id);")
    lines.append("")

    # Create relationships
    if graph.graph is not None and graph.graph.number_of_edges() > 0:
        lines.append(f"// Create inter-cluster relationships")
        for u, v, data in graph.graph.edges(data=True):
            weight = data.get("weight", 0.0)
            lines.append(
                f"MATCH (a:{node_label} {{cluster_id: {u}}}), "
                f"(b:{node_label} {{cluster_id: {v}}}) "
                f"CREATE (a)-[:SIMILAR_TO {{weight: {weight:.4f}}}]->(b);"
            )

    script = "\n".join(lines)

    with open(path, "w") as f:
        f.write(script)

    log.info("Cypher script written to %s (%d lines)", path, len(lines))
    return script


def gap_report_to_neo4j(
    gaps: list,
    supply_graph,
    uri: str = "bolt://localhost:7687",
    auth: Tuple[str, str] = ("neo4j", "neo4j"),
    database: str = "neo4j",
    label_prefix: str = "MF",
) -> dict:
    """
    Write coverage gap analysis results to Neo4j.

    Creates Gap nodes linked to their nearest supply cluster with
    a NEAREST_MATCH relationship showing the similarity score.

    Parameters
    ----------
    gaps : list[GapCluster]
        Output from SemanticGraph.coverage_gaps().
    supply_graph : SemanticGraph
        The fitted supply SemanticGraph.
    uri : str
        Neo4j Bolt URI.
    auth : tuple
        Username and password.
    database : str
        Target database.
    label_prefix : str
        Prefix for node labels.

    Returns
    -------
    dict
        Summary of nodes and relationships created.
    """
    try:
        from neo4j import GraphDatabase
    except ImportError:
        raise ImportError("Neo4j driver not installed. Run: pip install neo4j")

    driver = GraphDatabase.driver(uri, auth=auth)
    gap_label = f"{label_prefix}_Gap"
    supply_label = f"{label_prefix}_Cluster"
    nodes_created = 0
    rels_created = 0

    with driver.session(database=database) as session:
        for gap in gaps:
            props = {
                "cluster_id": gap.id,
                "size": gap.size,
                "volume": gap.volume,
                "top_terms": gap.top_terms[:10],
                "label": gap.top_terms[0] if gap.top_terms else f"Gap {gap.id}",
                "nearest_similarity": gap.nearest_similarity,
            }
            session.run(f"CREATE (n:{gap_label} $props)", props=props)
            nodes_created += 1

            # Link to nearest supply cluster if supply graph is loaded
            if gap.nearest_supply:
                session.run(
                    f"""
                    MATCH (g:{gap_label} {{cluster_id: $gap_id}})
                    MATCH (s:{supply_label}) WHERE s.label STARTS WITH $nearest
                    WITH g, s LIMIT 1
                    CREATE (g)-[:NEAREST_MATCH {{similarity: $sim}}]->(s)
                    """,
                    gap_id=gap.id,
                    nearest=gap.nearest_supply[:20],
                    sim=gap.nearest_similarity,
                )
                rels_created += 1

    driver.close()

    summary = {
        "gap_nodes_created": nodes_created,
        "nearest_match_rels": rels_created,
        "gap_label": gap_label,
    }
    log.info("Gap report exported: %d gaps, %d links", nodes_created, rels_created)
    return summary

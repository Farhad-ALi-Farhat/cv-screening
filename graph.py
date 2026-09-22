"""LangGraph wiring for the CV screening pipeline.

    START -> load_jd -> parse_jd -> convert_cvs -> match_candidates -> score_candidates -> build_report -> END
"""

from langgraph.graph import END, START, StateGraph

from nodes import (
    build_report_node,
    convert_cvs_node,
    load_jd_node,
    match_candidates_node,
    parse_jd_node,
    score_candidates_node,
)
from schemas import ScreeningState


def build_graph():
    graph = StateGraph(ScreeningState)

    graph.add_node("load_jd", load_jd_node)
    graph.add_node("parse_jd", parse_jd_node)
    graph.add_node("convert_cvs", convert_cvs_node)
    graph.add_node("match_candidates", match_candidates_node)
    graph.add_node("score_candidates", score_candidates_node)
    graph.add_node("build_report", build_report_node)

    graph.add_edge(START, "load_jd")
    graph.add_edge("load_jd", "parse_jd")
    graph.add_edge("parse_jd", "convert_cvs")
    graph.add_edge("convert_cvs", "match_candidates")
    graph.add_edge("match_candidates", "score_candidates")
    graph.add_edge("score_candidates", "build_report")
    graph.add_edge("build_report", END)

    return graph.compile()

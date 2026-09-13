from src.retrieval.cross_rag.run_cross_rag import run_cross_retrieval

"""
    These tests doesn't evaluate the performance of the cross rag pipeline.
    It just tests if the function is working without any error
"""

def test_cross_rag_without_generation():
    result = run_cross_retrieval("Wie viele Punkte gab die Verteidigung der Panthers ab?", "de")
    print(result)

def test_cross_rag_with_generation():
    result = run_cross_retrieval("Wie viele Punkte gab die Verteidigung der Panthers ab?", "de", generate=True)
    print(result)
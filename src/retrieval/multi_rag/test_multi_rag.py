from src.retrieval.multi_rag.run_multi_rag import run_multi_retrieval

"""
    These tests doesn't evaluate the performance of the multi rag pipeline.
    It just tests if the function is working without any error
"""

def test_multi_rag_without_generation():
    result = run_multi_retrieval("Wie viele Punkte gab die Verteidigung der Panthers ab?", "de")
    print(result)

def test_multi_rag_with_generation():
    result = run_multi_retrieval("Wie viele Punkte gab die Verteidigung der Panthers ab?", "de", generate=True)
    print(result)
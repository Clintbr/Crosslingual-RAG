from src.retrieval.t_rag.run_t_rag import run_trag_retrieval

"""
    These tests doesn't evaluate the performance of the translate rag pipeline.
    It just tests if the function is working without any error
"""

def test_t_rag_without_generation():
    result = run_trag_retrieval("Wie viele Punkte gab die Verteidigung der Panthers ab?", "de", "fr")
    print(result)

def test_t_rag_with_generation():
    result = run_trag_retrieval("Wie viele Punkte gab die Verteidigung der Panthers ab?", "de", "en", generate=True)
    print(result)
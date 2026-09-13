from src.retrieval.mono_rag.run_mono_rag import run_mono_retrieval

"""
    These tests doesn't evaluate the performance of the mono rag pipeline.
    It just tests if the function is working without any error
"""

def test_mono_rag_without_generation():
    result = run_mono_retrieval("Wie viele Punkte gab die Verteidigung der Panthers ab?", "de")
    print(result)

def test_mono_rag_with_generation():
    result = run_mono_retrieval("Wie viele Punkte gab die Verteidigung der Panthers ab?", "de", generate=True)
    print(result)
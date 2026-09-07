from src.retrieval.cross_rag.run_cross_rag import run_cross_retrieval


def test_t_rag_without_generation():
    result = run_cross_retrieval("Was wurde die Hydraulikpumpe ersetzt?", "de")
    print(result)

def test_t_rag_with_generation():
    result = run_cross_retrieval("Was wurde die Hydraulikpumpe ersetzt?", "de", generate=True)
    print(result)
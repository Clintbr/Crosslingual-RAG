from src.retrieval.t_rag.run_t_rag import run_trag_retrieval


def test_t_rag_without_generation():
    result = run_trag_retrieval("Was wurde die Hydraulikpumpe ersetzt?", "de", "fr")
    print(result)

def test_t_rag_with_generation():
    result = run_trag_retrieval("Was wurde die Hydraulikpumpe ersetzt?", "de", "fr", generate=True)
    print(result)
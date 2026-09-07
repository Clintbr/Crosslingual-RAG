from src.retrieval.multi_rag.run_multi_rag import run_multi_retrieval


def test_t_rag_without_generation():
    result = run_multi_retrieval("Wann wurde die Hydraulikpumpe ersetzt?", "de")
    print(result)

def test_t_rag_with_generation():
    result = run_multi_retrieval("Wann wurde die Hydraulikpumpe ersetzt?", "de", generate=True)
    print(result)
from src.retrieval.mono_rag.run_mono_rag import run_mono_retrieval


def test_t_rag_without_generation():
    result = run_mono_retrieval("Wann wurde die Hydraulikpumpe ersetzt?", "de")
    print(result)

def test_t_rag_with_generation():
    result = run_mono_retrieval("Wann wurde die Hydraulikpumpe ersetzt?", "de", generate=True)
    print(result)
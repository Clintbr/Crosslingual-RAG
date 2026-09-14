"""
 Here is the central for evaluating our rag pipeline. Each method from saving results to deep evaluation and
 like ragas with 'Llm as a Judge' are launched from here. After this Process to run sucessfully till the end,
 the evaluation for our rag strategies is considered finished.
"""

from src.config import (
    PREPARED_QUESTIONS_DIRECTORY, RETRIEVAL_QUALITY_RESULTS_DIRECTORY, RETRIEVAL_HARDWARE_RESULTS_DIRECTORY)
from src.evaluating.analysis.analyser import trigger_analyse_and_visualise
from src.evaluating.hardware.save_hardware_results import save_hardware_metrics
from src.evaluating.ragas.ragas_eval import evaluate_ragas_pipeline
from src.evaluating.ragas.run_rag_pipeline import run_pipeline
from src.utils.resolve_path import resolve_project_path

question_dir = resolve_project_path(PREPARED_QUESTIONS_DIRECTORY)
result_dir = resolve_project_path(RETRIEVAL_QUALITY_RESULTS_DIRECTORY)
hardware_dir = resolve_project_path(RETRIEVAL_HARDWARE_RESULTS_DIRECTORY)

pipelines = [
    # crossRAG
    {"method": "cross","input": f"{question_dir}/crossRAG/questions_de.json","store": f"{result_dir}/crossRAG/store/questions_de.json", "output": f"{result_dir}/crossRAG/questions_de_a.csv", "hardware": f"{hardware_dir}/crossRAG/questions_de_a.csv",},
    {"method": "cross","input": f"{question_dir}/crossRAG/questions_en.json","store": f"{result_dir}/crossRAG/store/questions_en.json", "output": f"{result_dir}/crossRAG/questions_en_a.csv", "hardware": f"{hardware_dir}/crossRAG/questions_en_a.csv",},
    {"method": "cross","input": f"{question_dir}/crossRAG/questions_fr.json","store": f"{result_dir}/crossRAG/store/questions_fr.json", "output": f"{result_dir}/crossRAG/questions_fr_a.csv", "hardware": f"{hardware_dir}/crossRAG/questions_fr_a.csv",},
    # monoRAG
    {"method": "mono","input": f"{question_dir}/monoRAG/questions_de.json","store": f"{result_dir}/monoRAG/store/questions_de.json", "output": f"{result_dir}/monoRAG/questions_de.csv", "hardware": f"{hardware_dir}/monoRAG/questions_de.csv",},
    {"method": "mono","input": f"{question_dir}/monoRAG/questions_en.json","store": f"{result_dir}/monoRAG/store/questions_en.json", "output": f"{result_dir}/monoRAG/questions_en.csv", "hardware": f"{hardware_dir}/monoRAG/questions_en.csv",},
    {"method": "mono","input": f"{question_dir}/monoRAG/questions_fr.json","store": f"{result_dir}/monoRAG/store/questions_fr.json", "output": f"{result_dir}/monoRAG/questions_fr.csv", "hardware": f"{hardware_dir}/monoRAG/questions_fr.csv",},
    # multiRAG
    {"method": "multi","input": f"{question_dir}/multiRAG/questions_de_1.json","store": f"{result_dir}/multiRAG/store/questions_de_1.json", "output": f"{result_dir}/multiRAG/questions_de_1.csv", "hardware": f"{hardware_dir}/multiRAG/questions_de_1.csv",},
    {"method": "multi","input": f"{question_dir}/multiRAG/questions_en_1.json","store": f"{result_dir}/multiRAG/store/questions_en_1.json", "output": f"{result_dir}/multiRAG/questions_en_1.csv", "hardware": f"{hardware_dir}/multiRAG/questions_en_1.csv",},
    {"method": "multi","input": f"{question_dir}/multiRAG/questions_fr_1.json","store": f"{result_dir}/multiRAG/store/questions_fr_1.json", "output": f"{result_dir}/multiRAG/questions_fr_1.csv", "hardware": f"{hardware_dir}/multiRAG/questions_fr_1.csv",},
    # tRAG
    {"method": "trag","input": f"{question_dir}/tRAG/questions_de_context_en.json","store": f"{result_dir}/tRAG/store/questions_de_context_en.json", "output": f"{result_dir}/tRAG/questions_de_context_en.csv", "hardware": f"{hardware_dir}/tRAG/questions_de_context_en.csv",},
    {"method": "trag","input": f"{question_dir}/tRAG/questions_de_context_fr.json","store": f"{result_dir}/tRAG/store/questions_de_context_fr.json", "output": f"{result_dir}/tRAG/questions_de_context_fr.csv", "hardware": f"{hardware_dir}/tRAG/questions_de_context_fr.csv",},
    {"method": "trag","input": f"{question_dir}/tRAG/questions_en_context_de.json","store": f"{result_dir}/tRAG/store/questions_en_context_de.json", "output": f"{result_dir}/tRAG/questions_en_context_de.csv", "hardware": f"{hardware_dir}/tRAG/questions_en_context_de.csv",},
    {"method": "trag","input": f"{question_dir}/tRAG/questions_en_context_fr.json","store": f"{result_dir}/tRAG/store/questions_en_context_fr.json", "output": f"{result_dir}/tRAG/questions_en_context_fr.csv", "hardware": f"{hardware_dir}/tRAG/questions_en_context_fr.csv",},
    {"method": "trag","input": f"{question_dir}/tRAG/questions_fr_context_de.json","store": f"{result_dir}/tRAG/store/questions_fr_context_de.json", "output": f"{result_dir}/tRAG/questions_fr_context_de.csv", "hardware": f"{hardware_dir}/tRAG/questions_fr_context_de.csv",},
    {"method": "trag","input": f"{question_dir}/tRAG/questions_fr_context_en.json","store": f"{result_dir}/tRAG/store/questions_fr_context_en.json", "output": f"{result_dir}/tRAG/questions_fr_context_en.csv", "hardware": f"{hardware_dir}/tRAG/questions_fr_context_en.csv",},
]

dataset = [
    # crossRAG
    {"method": "cross","input":f"{result_dir}/crossRAG/questions_de_a.csv", "hardware": f"{hardware_dir}/crossRAG/questions_de_a.csv",},
    {"method": "cross","input":f"{result_dir}/crossRAG/questions_en_a.csv", "hardware": f"{hardware_dir}/crossRAG/questions_en_a.csv",},
    {"method": "cross","input":f"{result_dir}/crossRAG/questions_fr_a.csv", "hardware": f"{hardware_dir}/crossRAG/questions_fr_a.csv",},
    # monoRAG
    {"method": "mono","input":f"{result_dir}/monoRAG/questions_de.csv", "hardware": f"{hardware_dir}/monoRAG/questions_de.csv",},
    {"method": "mono","input":f"{result_dir}/monoRAG/questions_en.csv", "hardware": f"{hardware_dir}/monoRAG/questions_en.csv",},
    {"method": "mono","input":f"{result_dir}/monoRAG/questions_fr.csv", "hardware": f"{hardware_dir}/monoRAG/questions_fr.csv",},
    # multiRAG
    {"method": "multi","input":f"{result_dir}/multiRAG/questions_de_1.csv", "hardware": f"{hardware_dir}/multiRAG/questions_de_1.csv",},
    {"method": "multi","input":f"{result_dir}/multiRAG/questions_en_1.csv", "hardware": f"{hardware_dir}/multiRAG/questions_en_1.csv",},
    {"method": "multi","input":f"{result_dir}/multiRAG/questions_fr_1.csv", "hardware": f"{hardware_dir}/multiRAG/questions_fr_1.csv",},
    # tRAG
    {"method": "trag","input":f"{result_dir}/tRAG/questions_de_context_en.csv", "hardware": f"{hardware_dir}/tRAG/questions_de_context_en.csv",},
    {"method": "trag","input":f"{result_dir}/tRAG/questions_de_context_fr.csv", "hardware": f"{hardware_dir}/tRAG/questions_de_context_fr.csv",},
    {"method": "trag","input":f"{result_dir}/tRAG/questions_en_context_de.csv", "hardware": f"{hardware_dir}/tRAG/questions_en_context_de.csv",},
    {"method": "trag","input":f"{result_dir}/tRAG/questions_en_context_fr.csv", "hardware": f"{hardware_dir}/tRAG/questions_en_context_fr.csv",},
    {"method": "trag","input":f"{result_dir}/tRAG/questions_fr_context_de.csv", "hardware": f"{hardware_dir}/tRAG/questions_fr_context_de.csv",},
    {"method": "trag","input":f"{result_dir}/tRAG/questions_fr_context_en.csv", "hardware": f"{hardware_dir}/tRAG/questions_fr_context_en.csv",},
]

if __name__ == "__main__":
    """
    for pipeline in pipelines:
        # run rag methods and save results(question, answer, time, ...) in .json files
        run_pipeline(
            method_name=pipeline["method"],
            input_json=pipeline["input"],
            output_json=pipeline["store"]
        )
        # evaluate according to ragas process the rag results and save it in .csv files
        # id,user_input,reference,response,retrieved_contexts,error_phase,error_type,error_message,error_status_code,faithfulness,answer_relevancy,answer_correctness,context_precision,context_recall,context_relevance
        evaluate_ragas_pipeline(
            input_json=pipeline["store"],
            output_csv=pipeline["output"],
        )

        # save hardware results
        # question_id,avg_ram_mb,peak_ram_mb,avg_cpu_percent,peak_cpu_percent,rag_strategy,phase
        save_hardware_metrics(
            input_json=pipeline["store"],
            output_csv=pipeline["hardware"],
        )
        """
    # analyse, visualize and save as pdfs and pngs
    trigger_analyse_and_visualise(
        dataset=dataset
    )
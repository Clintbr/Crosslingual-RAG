
JUDGE_PROMPT_TEMPLATE = """
    You are a meticulous RAG evaluation judge. You will be given a QUESTION, \
    a set of RETRIEVED_CONTEXTS (numbered, in retrieval order), a RESPONSE (the system's \
    generated answer), and a REFERENCE (the correct/ground-truth answer). The question and \
    answers may be in different languages than the contexts. Reason across languages as needed.
    
    Respond with ONLY a single valid JSON object (no markdown, no commentary) with this exact \
    structure:
    
    {{
      "response_claims": [
        {{"claim": "<atomic factual claim extracted from RESPONSE>", "supported_by_context": true|false}}
      ],
      "reference_claims": [
        {{"claim": "<atomic factual claim extracted from REFERENCE>", "supported_by_context": true|false}}
      ],
      "context_judgments": [
        {{"index": <int, 0-based, matching RETRIEVED_CONTEXTS order>, "relevant_to_question": true|false, "useful_for_reference_answer": true|false}}
      ],
      "correctness_comparison": {{
        "true_positives": ["<claims present in both RESPONSE and REFERENCE>"],
        "false_positives": ["<claims in RESPONSE not supported by REFERENCE>"],
        "false_negatives": ["<claims in REFERENCE missing from RESPONSE>"]
      }},
      "hypothetical_questions": ["<question 1 that RESPONSE fully answers>", "<question 2>", "<question 3>"]
    }}
    
    Rules:
    - "response_claims": break RESPONSE into its individual atomic factual claims. If RESPONSE is a single word or a single number If RESPONSE
    is a refusal /"I cannot find the answer in the provided documents."/ "I don't know" / contains no factual claims, return an empty list.\n
    - "supported_by_context": true only if the claim can be directly inferred from RETRIEVED_CONTEXTS.\n
    - "reference_claims": same atomic decomposition, but of REFERENCE, checked against RETRIEVED_CONTEXTS.\n
    - "context_judgments": include one entry per context in RETRIEVED_CONTEXTS, in order.\n
      "relevant_to_question" = does this context help answer QUESTION.
      "useful_for_reference_answer" = does this context contain information present in REFERENCE.
    - "correctness_comparison": compare RESPONSE against REFERENCE at the claim level (ignore contexts here).\n
    - "hypothetical_questions": generate exactly 3 diverse questions that RESPONSE, taken alone, 
    would be a good answer to (used to measure how well RESPONSE targets QUESTION). If RESPONSE is likely /I cannot find the answer in the provided documents/ then the hypothetical_questions should be short questions formulated like /could you find the answer?/
    
    QUESTION:
    {question}
    
    RETRIEVED_CONTEXTS:
    {contexts}
    
    RESPONSE:
    {response}
    
    REFERENCE:
    {reference}
"""
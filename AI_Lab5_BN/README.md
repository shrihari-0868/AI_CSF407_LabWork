# Building and Learning a Bayesian Network with an LLM: completed notebook

## Files

| File | What it is |
|---|---|
| `llm_bn_completed.ipynb` | The course notebook with every exercise completed, executed top to bottom with all outputs |
| `llm_bn_completed.html` | The same notebook rendered as HTML; opens in any browser |
| `llm_responses.json` | Exact record of every prompt sent to the LLM and its response (with generation time) |

## What was completed

1. **Generated code approved, executed and validated** (inference, MLE and BDeu). Every LLM program was reviewed line by line, executed safely and checked against the specification.
2. **Stronger validation**: a stricter static check (it catches `pd.read_csv` and overwriting `data`) and a CPT-level semantic validator that checks every probability by named state. A prior-sensitivity test checks that the BDeu validator can tell ESS = 10 from MLE and from ESS = 5.
3. **Revise step**: an automatic LLM repair loop that feeds the error back to the model, followed by human-revised programs that pass all tests.
4. **HW exercise (change the query)**:
   - For each of P(S=1|W=1), P(C=1|W=1) and P(R=1|W=1,S=0): the query variable and evidence, a qualitative prediction with reasoning, the LLM-generated code with inspection notes, and a comparison with `VariableElimination` and brute-force enumeration.
   - Results: 0.4278 (prior 0.30), 0.5746 (prior 0.50) and 0.9922 (was 0.7048). All three predictions were confirmed.
5. **Classroom exercise, CPT explanation**: the LLM's explanation was checked against pgmpy's column convention. It swapped the two middle columns and 3 of its 4 probabilities were wrong.
6. **Classroom exercise, different datasets**: estimates over several seeds, 100 datasets per sample size, standard deviation compared with the theory sqrt(p(1-p)/N_c), a box plot, and the LLM's explanation critiqued.
7. **Rubric** applied to every program, with conclusions (end of the notebook).

## Environment

- Python 3.13, pgmpy 1.1.2, transformers 5.17, torch 2.14 (CPU only)
- LLM: `Qwen/Qwen2.5-Coder-1.5B-Instruct`, greedy decoding (`do_sample=False`)
- Because responses are cached in `llm_responses.json`, re-running the notebook in the same folder reproduces the outputs exactly without regenerating anything.

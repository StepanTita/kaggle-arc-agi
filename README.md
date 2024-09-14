# ARC AGI Challenge Ideas:

- This is a mat2mat task. We get some matrix + examples as an input, and have to produce 2 options for output.
- The task is to find the rule that transforms the input matrix into the output matrix.
- This can be done through code generation, or through a more general approach.

* Chain of thoughts
* Combine Visual models with LLM
* CLIP for similar embeddings
* Use one LLM to generate hypothesis, use another LLMs to test them
* LLMs generate proofs why they believe their solution is correct
* Use judge LLM to choose best solution
* Return result that is agreed by the majority of LLMs
* Set of the instructions how input is generated (in the prompts)
* Iterative generation of the steps to go from state A to state B
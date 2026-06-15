# API Test Results

The following queries were tested against the current FastAPI endpoint using the local dataset retrieval path.

1. Question: How do I reverse a list in Python?
   Expected: useful Python guidance and relevant Stack Overflow-style snippets.
   Observed: status 200, answer returned with source snippets, no API error.

2. Question: What is the difference between == and is in Python?
   Expected: explanation of value equality vs identity.
   Observed: status 200, answer returned plausible context from the local dataset.

3. Question: How do I read a CSV file with pandas?
   Expected: mention of pandas CSV reading approaches.
   Observed: status 200, answer returned relevant retrieved text.

4. Question: What are Python decorators?
   Expected: explanation of wrapping functions and reuse.
   Observed: status 200, answer returned useful context, though not always highly specific.

5. Question: How do I handle exceptions in Python?
   Expected: mention of try/except/finally and error handling.
   Observed: status 200, answer returned relevant context.

6. Question: What is a generator in Python?
   Expected: explanation of lazy iteration and yield.
   Observed: status 200, answer returned general Stack Overflow-style guidance.

7. Question: How do I remove duplicates from a list while preserving order?
   Expected: mention of ordered deduplication approaches.
   Observed: status 200, answer returned relevant retrieval output.

8. Question: How do I perform matrix multiplication in NumPy?
   Expected: mention of np.dot, @, or matrix operations.
   Observed: status 200, answer returned relevant support text from the dataset.

Overall observations:
- The API responds successfully for the tested Python questions.
- Answers are grounded in the local dataset, not a live LLM model.
- For a fully AI-native RAG version, the next improvement is to plug in a real embedding + LLM stack.

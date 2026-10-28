Bug review and fixes
====================

The repository README describes 40 score checks (35 component checks and 5
integration checks). I found and corrected the following concrete defects in
the checked-in implementation. The tests and GitHub workflow were left intact.

Model configuration and architecture
------------------------------------
1. Default d_model was 769, which is not divisible by 12. Changed it to 768.
2. Default vocabulary size was 50256 instead of 50257. Corrected the default.
3. EOS_TOKEN_ID was outside the vocabulary and disagreed with GPT-2. Set it to 50256.
4. UNK_TOKEN_ID incorrectly pointed at the end-of-text token. Set it to 1.
5. The causal mask blocked past tokens and allowed future tokens. Reversed the mask.
6. Attention scores were not divided by sqrt(head_dim). Added the required scaling.
7. Attention softmax normalized over query positions. It now normalizes over keys.
8. Query, key, and value projections shared one Linear layer. They are independent.
9. Attention heads were concatenated over the sequence axis. They now join on features.
10. Sinusoidal position vectors used cosine for both even and odd dimensions. Fixed sine.
11. Odd-width sinusoidal vectors had an incompatible cosine assignment. Added a size slice.
12. Learned position embeddings did not guard invalid offsets/context lengths. Added checks.
13. Embedding vectors multiplied token and position vectors. They now add them.
14. Token IDs were wrapped modulo vocabulary size, silently changing invalid IDs. Removed it.
15. LMHead projected vocab_size to d_model, reversing its input/output dimensions. Fixed it.
16. LayerNorm normalized across batch/time instead of the final feature axis. Fixed the axis.
17. ResidualConnection returned x + x and discarded the sublayer. Restored dropout(sublayer).
18. FeedForward ignored the configured expansion ratio. It now uses configured d_ffn.
19. TransformerBlock applied FFN before attention. Restored pre-norm attention then FFN order.
20. Rotary embedding accepted odd dimensions and lacked offset handling. Added validation/slicing.
21. FullTransformerModel omitted the explicit config for its block stack. It now passes it.
22. FullTransformerModel did not report its layer count or tie configured weights. Fixed both.

Tokenizer and text generation
-----------------------------
23. Encoding ignored add_bos. It now prepends BOS when requested.
24. Decoding discarded tokens based only on sequence length. It strips actual boundary IDs.
25. Decoding stripped boundary tokens even when skip_special_tokens=False. Fixed the option.
26. Vocabulary pad_token_id was -1 despite PAD being stored at ID 0. Corrected the field.
27. Vocabulary instances shared mutable class-level maps and polluted one another. Removed them.
28. Vocabulary could assign two tokens to one ID. Duplicate-ID assignments now fail clearly.
29. BPE sorted merges lexically rather than respecting their declared rank. Uses merge rank.
30. BPE merged only the first matching pair occurrence in a word. Merges all occurrences.
31. Top-k sampling retained the lowest logits. It now retains the highest logits.
32. Top-p sampling used vocabulary order rather than probability order. It sorts then scatters.
33. Temperature scaled probabilities and divided by zero. It now scales logits and validates.
34. Sampler's zero-temperature path sampled stochastically. It now selects argmax.
35. Repetition penalty subtracted a constant. It now uses sign-aware multiplicative penalty.
36. Autoregressive generation prepended each new token, reversing output order. Appends instead.
37. StopCriteria stopped only after exceeding max length. It stops at the configured limit.
38. KVCache overwrote prior entries instead of appending sequence positions. It concatenates.
39. The inference model entered training mode for generation. It now uses eval mode/device.
40. Inference generation ignored top_k/top_p/repetition settings. It now uses Sampler.
41. Inference mishandled tuple model outputs and empty prompts. Both cases are handled.
42. Inference accepted negative temperature and ignored the configured seed. Added checks/seed.
43. InferenceEngine returned fabricated continuation text after failures. Removed fabricated text.

Serving, utilities, and evaluation
----------------------------------
44. to_device always moved tensors to CPU. It now honors the requested device.
45. ServingPipeline passed a Python list to the model without batch/device conversion. Fixed it.
46. ServingPipeline ignored max_new_tokens and did not autoregressively generate. Added a loop.
47. Completion responses returned token lists as text. They now return decoded completion text.
48. The generate-only API path discarded its result and substituted made-up token IDs. Fixed it.
49. The server used HTTPServer despite claiming threaded request handling. Uses ThreadingHTTPServer.
50. Cross-entropy used view on potentially non-contiguous tensors. It now uses reshape.
51. Forward benchmarking accepted zero measured runs and then failed on empty statistics. Validates.
52. Batch chunking accepted zero/negative chunk sizes. It now validates the size.
53. DynamicBatcher accepted unusable batch sizes and negative wait durations. It validates them.
54. ModelConfig accepted nonpositive dimensions and invalid dropout/context values. Added checks.
55. InferenceConfig did not validate beam_width or seed types. Added checks.

Validation status
-----------------
The environment could not resolve PyPI, so `pip install -r requirements.txt` could not
download torch. I did not run the test suite. Source changes were reviewed with
`git diff --check`; a Python syntax compilation is still needed in an environment
with `python3` available.

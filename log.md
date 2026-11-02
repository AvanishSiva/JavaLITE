JavaLite tokenizer comparision with GPT2 and QWEN2.5-coder

tokenizer         vocab     tokens  bytes/token
gpt2             50,257  1,037,506         2.46
qwen2.5-coder   151,665    545,785         4.67
javalite-16k     16,384    639,017         3.99

JavaLite's tokinzer did better than GPT2 considering it knows how to merge the common atanomy together. It is logically right for qwen to score more, for its vocab size.
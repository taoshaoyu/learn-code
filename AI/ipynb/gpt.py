from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelWithLMHead

gpt_config = AutoConfig.from_pretrained('openai-gpt')
gpt_tokenizer = AutoTokenizer.from_pretrained('openai-gpt')
gpt_model = AutoModel.from_pretrained('openai-gpt')
gpt_model_with_lmhead = AutoModelWithLMHead.from_pretrained('openai-gpt')
gpt_inputs = gpt_tokenizer("Hello, my dog is cute", return_tensors="pt")
gpt_output=gpt_model_with_lmhead.generate(gpt_inputs.input_ids)
gpt_output_text=gpt_tokenizer.batch_decode(gpt_output)
print(gpt_output_text)
a=gpt_model(**gpt_inputs)
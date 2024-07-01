from transformers import AutoModelForCausalLM, AutoTokenizer
gptj_tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6b')
gptj_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b")
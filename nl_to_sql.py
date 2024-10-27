from transformers import AutoModelForCausalLM, AutoTokenizer

model_name_or_path = "TheBloke/sqlcoder2-GPTQ"

model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

def natural_language_to_sql(human_input, prompt_template, model=model, tokenizer=tokenizer):
    """Converts a natural language question to a SQL query using the model and tokenizer provided"""
    
    prompt_template = prompt_template.replace("{human_input}", human_input)

    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.5, do_sample=True, top_p=0.8, top_k=40, max_new_tokens=128)
    result = tokenizer.decode(output[0], skip_special_tokens=True)

    return result[result.find("SELECT"):]




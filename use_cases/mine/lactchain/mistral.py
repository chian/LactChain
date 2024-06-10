from __future__ import annotations

import sys, os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Tuple

def generate(MODEL:str, prompts:list[str]) -> Tuple[list[str], list[str]]: 

    model=AutoModelForCausalLM.from_pretrained(MODEL, device_map='auto')
    tokenizer=AutoTokenizer.from_pretrained(MODEL)

    tokenizer.pad_token=tokenizer.eos_token
    batch_encoding = tokenizer(prompts, return_tensors="pt", padding='longest')
    '''BatchEncoding: 
    Dict{
        input_ids:Tensor B x Seq_len
        attention_mask:Tensor B x Seq_len
        }
    '''
    batch_encoding = batch_encoding.to(model.device) # returns 
    model.eval()
    with torch.no_grad():
        '''Tensor of shape B, len(max_new_tokens + input_tokens)'''
        generated_tokens=model.generate(**batch_encoding, 
                                        num_return_sequences=1, 
                                        max_new_tokens=500)
    input_token_length = batch_encoding['input_ids'].shape[-1]
    processed_tokens=[output[input_token_length:] for output in generated_tokens] # output = row in tensor

    decoded_outputs = tokenizer.batch_decode(
            processed_tokens,
            skip_special_tokens=True,
        )
    
    return decoded_outputs

if __name__=="__main__": 
    import sys, os
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    MODEL='mistralai/Mistral-7B-Instruct-v0.3'

    test_messages=[
        '''<s>[INST] You are an intelligent agi in grid-world that is of size 5x5. 
        You may only make one of the following moves to navigate: [move_forward, move_left]
        Propose a sequence of moves that will allow you to explore or get you to the goal. 
        All of your output must be stored in a json in the following format, and nothing else: 
        {{
        'explain': //Your explanation and logic goes here//
        'moves': // Your sequence of moves goes here// 
        }} 
        YOU ARE NOT ALLOWED TO OUTPUT ANYTHING ELSE THAT DOES NOT STRICTLY ADHERE TO THE JSON FORMAT ABOVE.
        [/INST]''', 

        '''<s>[INST] You are an intelligent agi in grid-world that is of size 4x4. 
        You may only make one of the following moves to navigate: [move_forward, move_left]
        Propose a sequence of moves that will allow you to explore or get you to the goal. Your output must be in the format: 
        {{'moves': // Your sequence of moves goes here// }} [/INST]''', 

        '''<s>[INST] You are an intelligent agi in grid-world that is of size 10x10. 
        You may only make one of the following moves to navigate: [move_forward, move_left]
        Propose a sequence of moves that will allow you to explore or get you to the goal. Your output must be in the format: 
        {{'moves': // Your sequence of moves goes here// }} [/INST]'''
    ]
    decoded_outputs, processed_outputs=generate(MODEL, test_messages)

    breakpoint()

    # model=AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3', device_map='auto')
    # tokenizer=AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')

    # # prompt=['''<s> [INST] Instruction [/INST] Model answer</s>''']

    # input_texts = [
    # "You are in grid_world. Can you come up with a sequence of moves to get to the end?",
    # "Who wrote 'Pride and Prejudice'?",
    # "Write a short story for me in this format: {{story:' // your story goes in here //'}} "
    # ]

    # tokenizer.pad_token=tokenizer.eos_token

    # batch_encoding = tokenizer(input_texts, return_tensors="pt", padding='longest')

    # model.eval()
    # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # model.to(device)
    # batch_encoding = batch_encoding.to(model.device)
    # with torch.no_grad():
    #     generated_text=model.generate(**batch_encoding, 
    #                                 num_return_sequences=1, 
    #                                 max_new_tokens=500)

    # decoded_outputs = tokenizer.batch_decode(
    #         generated_text,
    #         skip_special_tokens=True,
    #     )


    # messages = [
    # {"role": "user", "content": "What is your favourite condiment?"},
    # {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    # {"role": "user", "content": "Do you have mayonnaise recipes?"}
    # ]

    # tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    # print(tokenizer.decode(tokenized_chat[0]))

    # test_messages=[
    #     '''<s>[INST] You are an intelligent agi in grid-world that is of size 5x5. 
    #     You may only make one of the following moves to navigate: [move_forward, move_left]
    #     Propose a sequence of moves that will allow you to explore or get you to the goal. Your output must be in the format: 
    #     {{'moves': // Your sequence of moves goes here// }} [/INST]''', 

    #     '''<s>[INST] You are an intelligent agi in grid-world that is of size 4x4. 
    #     You may only make one of the following moves to navigate: [move_forward, move_left]
    #     Propose a sequence of moves that will allow you to explore or get you to the goal. Your output must be in the format: 
    #     {{'moves': // Your sequence of moves goes here// }} [/INST]''', 

    #     '''<s>[INST] You are an intelligent agi in grid-world that is of size 10x10. 
    #     You may only make one of the following moves to navigate: [move_forward, move_left]
    #     Propose a sequence of moves that will allow you to explore or get you to the goal. Your output must be in the format: 
    #     {{'moves': // Your sequence of moves goes here// }} [/INST]'''
    # ]

    # batch_inputs=tokenizer(test_messages, padding='longest', return_tensors='pt').to(model.device)
    # outputs=model.generate(**batch_inputs, num_return_sequences=1, max_new_tokens=500)
    # batch_decoded=tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # breakpoint()
    # mistral_models_path=os.getcwd()
    # breakpoint()
    # tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tokenizer.model.v3")
    # model = Transformer.from_folder(mistral_models_path)
    # completion_request = ChatCompletionRequest(messages=[UserMessage(content="Explain Machine Learning to me in a nutshell.")])
    # tokens = tokenizer.encode_chat_completion(completion_request).tokens
    # out_tokens, _ = generate([tokens], model, max_tokens=64, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
    # result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

    # breakpoint()



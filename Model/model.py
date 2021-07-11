from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Config
import torch  
import pandas as pd
def preprocess(abstract, title = None):
    if type(abstract) == str:
        abstract = "summarize: "+ abstract + "</s>"
        content = tokenizer.encode_plus(abstract.split(), max_length=300, padding=True, return_tensors='pt', truncation=True)
        if title:
            title = title + "</s>"    
            labels = tokenizer.encode_plus(title.split(), return_tensors='pt', max_length=20, padding=True, truncation=True)#.to(device)
            return content["input_ids"], labels["input_ids"]
        return content["input_ids"]
    else:
        abstract = abstract.apply(lambda x: ("summarize: "+ x + "</s>"))
        content = tokenizer.batch_encode_plus(abstract.tolist(), max_length=300, padding=True, return_tensors='pt', truncation=True)#.to(device)
        if title:
            title = title.apply(lambda x:  x + "</s>")    
            labels = tokenizer.batch_encode_plus(title.tolist(), return_tensors='pt', max_length=20, padding=True, truncation=True)#.to(device)

            return content["input_ids"], labels["input_ids"]
        else:
            return content["input_ids"]

tokenizer = AutoTokenizer.from_pretrained("t5-small", is_split_into_words=True)
#configT5 = T5Config(n_positions = 64, decoder_start_token_id = tokenizer.pad_token_id)  
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")#, config = configT5)#.to(device)
# state_dict = torch.load('model_weights_3_2.pth', map_location=torch.device('cpu'))
#state_dict = torch.load('weights.pth', map_location=torch.device('cpu'))
#model.load_state_dict(state_dict, strict=False)
model.eval()


def pred(text):
    text = pd.DataFrame(list([text]), columns=['abstract'])
    X_test = preprocess(text['abstract'])
    eval_outputs = model.generate(
        X_test.clone().detach(),#.unsqueeze(0), 
        num_return_sequences=3,
        max_length=25, 
        min_length=10, 
        length_penalty=10,  
        num_beams=8, 
        early_stopping=True)
    pred = tokenizer.batch_decode(eval_outputs, skip_special_tokens=True)

    ans = []
    for i, title in enumerate(pred):
        ans.append([i,title])

    pred = dict({'result':ans})
    print(pred)
    return pred

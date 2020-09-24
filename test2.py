import torch
from tqdm import tqdm
import nltk
import transformers as tr

import load_data as ld
from train2 import LSTMModel, LSTMEncoder, LSTMDecoder


def test_lstm(data, model, tokenizer, save_path, device="cpu"):
    """
    Run trained model on test data.
    Save outputs to a file.

    Parameters
    ----------
    data : [(str,str)]
        Test pairs of normal and
        simplified sentences of texts.
    model: nn.Module
        Model to test.
    tokenizer : ?
        Corresponding tokenizer.
    save_path : str
        Path to save resulting sentences to.
    device : str
        Device to use.
        "cpu" for CPU, "cuda" for GPU.
    """

    model.to(device)
    model.eval()

    generated_sents = []

    i = 0
    for (normal, simple) in tqdm(data):
        inputs_prepared = tokenizer(normal, max_length=70-2, padding='max_length', truncation=True, return_tensors='pt')["input_ids"]
        inputs_prepared = inputs_prepared.to(device)

        outputs = model(inputs_prepared)
        generated = torch.argmax(outputs, dim=2)
        #print(outputs.shape)
        #7/0

        generated = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        generated = generated.replace("\r", " ")
        generated = generated.replace("\n", " ")
        generated_sents.append(generated)

        if i < 10:
            print(normal)
            print(generated)
            print(simple)
            print("\n")
        #else:
        #    break

        i += 1

    with open(save_path, "w") as out_file:
        for sent in generated_sents:
            out_file.write(sent)
            out_file.write("\n")


def test_enc_dec(data, model, tokenizer, save_path, device="cpu"):
    """
    Run trained model on test data.
    Save outputs to a file.

    Parameters
    ----------
    data : [(str,str)]
        Test pairs of normal and
        simplified sentences of texts.
    model: (nn.Module, nn.Module)
        Encoder and decoder.
    tokenizer : ?
        Corresponding tokenizer.
    save_path : str
        Path to save resulting sentences to.
    device : str
        Device to use.
        "cpu" for CPU, "cuda" for GPU.
    """

    #model.to(device)
    #model.eval()
    encoder, decoder = model
    encoder.to(device)
    decoder.to(device)

    generated_sents = []

    j = 0
    for (normal, simple) in tqdm(data):
        inputs_prepared = tokenizer(normal, max_length=70-2, padding='max_length', truncation=True, return_tensors='pt')["input_ids"]
        inputs_prepared = inputs_prepared.to(device).transpose(0,1)

        #outputs = model(inputs_prepared)
        enc_out = []
        hidden, cell = encoder.init_states(device, 1)
        for tok in inputs_prepared:
            tok = tok.unsqueeze(0)
            _, (hidden, cell) = encoder(tok, (hidden, cell))
        #curr_tok_id = torch.Tensor([tokenizer.bos_token_id]).to(device)
        curr_tok_id = inputs_prepared[0]
        curr_tok_id = curr_tok_id.unsqueeze(0)
        #curr_tok_id = tokenizer(tokenizer.bos_token, return_tensors='pt')["input_ids"][0].to(device).unsqueeze(0)
        out_ids = [curr_tok_id]
        _, cell = decoder.init_states(device, 1)
        #cell = cell.to(torch.int64)
        for i in range(70):
            #curr_tok_id = curr_tok_id.unqueeze(0)
            out, (hidden, cell) = decoder(curr_tok_id, (hidden, cell))
            curr_tok_id = torch.argmax(out, dim=2)
            out_ids.append(curr_tok_id)
        out_ids = torch.stack(out_ids)
        #print(generated.shape)
        generated = torch.argmax(out_ids, dim=2)
        #print(outputs.shape)
        #7/0

        generated = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        generated = generated.replace("\r", " ")
        generated = generated.replace("\n", " ")
        generated_sents.append(generated)

        if j < 10:
            print(j)
            print(normal)
            print(generated)
            print(simple)
            print("\n")
        else:
            break

        j += 1

    with open(save_path, "w") as out_file:
        for sent in generated_sents:
            out_file.write(sent)
            out_file.write("\n")



model_path = "/data/tuned_models/supervised/lstm_newsela_teacherf.pt"

#data = ld.load_wiki_sents("/data/data/wiki/sent_aligned_split/test")
#data = ld.load_newsela_sents("/data/data/newsela/V0V2/test")
data = ld.load_newsela_sents("/data/data/newsela/V0V2/train")[:1]
tokenizer = tr.BartTokenizer.from_pretrained("facebook/bart-base")

#model = LSTMModel(tokenizer)
#model.eval()
#model.load_state_dict(torch.load(model_path))
#for epoch in range(20):
#    print(epoch)
epoch = ""
encoder = LSTMEncoder(tokenizer, 1)
encoder.load_state_dict(torch.load(model_path+".enc"+str(epoch)))
encoder.eval()
decoder = LSTMDecoder(tokenizer, encoder, 1)
decoder.load_state_dict(torch.load(model_path+".dec"+str(epoch)))
decoder.eval()

test_enc_dec(data, (encoder, decoder), tokenizer, "/data/test_results/lstm_newsela.txt", device="cuda")
print("\n\n\n\n")

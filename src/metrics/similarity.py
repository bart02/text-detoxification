from nltk.translate.bleu_score import sentence_bleu
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings
from torch.nn.functional import cosine_similarity
import numpy as np
import torch

def calc_bleu(inputs, preds):
    results = []
    
    print('Calculating BLEU similarity')
    for i in range(len(inputs)):
        results.append(sentence_bleu([inputs[i]], preds[i]))

    return np.array(results)


def flair_sim(inputs, preds, batch_size = 16):
    print('Calculating flair embeddings similarity')
    sim = 0
    batch_size = 16
    inp_embed = []
    pred_embed = []

    embedder = FlairEmbeddings('news-forward')

    for i in range(0, len(inputs), batch_size):
        inp_part = [Sentence(sent) for sent in inputs[i:i + batch_size]]
        pred_part = [Sentence(sent) for sent in preds[i:i + batch_size]]

        inp_part = embedder.embed(inp_part)
        pred_part = embedder.embed(pred_part)

        for j in range(batch_size):
            if ((i + j) < len(inputs)):
                inp_sent_vec = torch.zeros(2048).cuda()
                pred_sent_vec = torch.zeros(2048).cuda()

                for k in range(len(inp_part[j])):
                    inp_sent_vec += inp_part[j][k].embedding
                inp_embed.append(inp_sent_vec.cpu() / (k + 1))

                for k in range(len(pred_part[j])):
                    pred_sent_vec += pred_part[j][k].embedding
                pred_embed.append(pred_sent_vec.cpu() / (k + 1))

    emb_sim = cosine_similarity(torch.stack(inp_embed), torch.stack(pred_embed))

    return emb_sim.numpy()

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm
import numpy as np


def caption_evaluate(predictions, targets, tokenizer, text_trunc_length):
    meteor_scores = []
    references = []
    hypotheses = []
    for gt, out in tqdm(zip(targets, predictions)):
        gt_tokens = tokenizer.tokenize(gt, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

        out_tokens = tokenizer.tokenize(out, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
        out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
        out_tokens = list(filter(('[SEP]').__ne__, out_tokens))

        references.append([gt_tokens])
        hypotheses.append(out_tokens)

        mscore = meteor_score([gt_tokens], out_tokens)
        meteor_scores.append(mscore)

    bleu2 = corpus_bleu(references, hypotheses, weights=(.5,.5))
    bleu4 = corpus_bleu(references, hypotheses, weights=(.25,.25,.25,.25))
    bleu2 *= 100
    bleu4 *= 100

    print('BLEU-2 score:', bleu2)
    print('BLEU-4 score:', bleu4)
    _meteor_score = np.mean(meteor_scores)
    _meteor_score *= 100
    print('Average Meteor score:', _meteor_score)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    rouge_scores = []

    references = []
    hypotheses = []

    for gt, out in tqdm(zip(targets, predictions)):
        rs = scorer.score(out, gt)
        rouge_scores.append(rs)

    print('ROUGE score:')
    rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores]) * 100
    rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores]) * 100
    rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores]) * 100
    print('rouge1:', rouge_1)
    print('rouge2:', rouge_2)
    print('rougeL:', rouge_l)
    return bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

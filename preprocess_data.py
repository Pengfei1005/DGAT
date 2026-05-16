import os
import json
import argparse

SPECIAL_TOKENS = {
    '<NULL>': 0,
    '<UNK>': 1,
    '<START>': 2,
    '<END>': 3,
}

def main(args):
    global input_vocab
    if args.dataset == 'LEVIR_CC':
        input_caption = './data/LEVIR_CC/LevirCCcaptions.json'
        input_vocab = ''
        output_vocab = 'vocab.json'
        save_path = './data/LEVIR_CC/'
    elif args.dataset == 'Dubai_CC':
        input_caption = './data/Dubai_CC/Dubai_caption.json'
        input_vocab = ''
        output_vocab = 'vocab.json'
        save_path = './data/Dubai_CC/'
    elif args.dataset == 'WHU_CDC':
        input_caption = './data/WHU_CDC/whuCCcaptions.json'
        input_vocab = ''
        output_vocab = 'vocab.json'
        save_path = './data/WHU_CDC/'
        
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path + 'tokens/'):
        os.makedirs(os.path.join(save_path + 'tokens/'))

    print('Loading Captions')
    if args.dataset == 'LEVIR_CC':
        with open(input_caption, 'r') as f:
            data = json.load(f)
        max_length = -1
        all_caption_tokens = []
        for img in data['images']:
            current_captions = []
            for caption in img['sentences']:
                assert len(caption['raw']) > 0, 'error: there is no caption for some images.'
                current_captions.append(caption['raw'])
            tokens_list = []
            for captions in current_captions:
                caption_tokens = tokenize(captions,
                                      add_start_token=True,
                                      add_end_token=True,
                                      punt_to_keep=[';', ','],
                                      punt_to_remove=['?', '.']
                                      )
                tokens_list.append(caption_tokens)
                max_length = max(max_length, len(caption_tokens))
            all_caption_tokens.append((img['filename'], tokens_list))

        print('Saving Captions')
        for img, token_list in all_caption_tokens:
            i = img.split('.')[0]
            token_lens = len(token_list)
            token_list = json.dumps(token_list)
            f = open(os.path.join(save_path + 'tokens/', i + '.txt'), 'w')
            f.write(token_list)
            f.close()

            if i.split('_')[0] == 'train':
                f = open(os.path.join(save_path + 'train' + '.txt'), 'a')
                for j in range(token_lens):
                    f.write(img + '-' + str(j) + '\n')
                f.close()
            elif i.split('_')[0] == 'val':
                f = open(os.path.join(save_path + 'val' + '.txt'), 'a')
                f.write(img + '\n')
                f.close()
            elif i.split('_')[0] == 'test':
                f = open(os.path.join(save_path + 'test' + '.txt'), 'a')
                f.write(img + '\n')
                f.close()

    print('Max_length of the dataset is', max_length)
    if input_vocab == '':
        print('Building Vocabulary')
        word_freq = building_vocab(all_caption_tokens, args.word_count_threshold)
    else:
        print('Loading Vocab')
        with open(input_vocab, 'r') as f:
            word_freq = json.load(f)
    if output_vocab  is not None:
        print('Saving Vocab')
        with open(os.path.join(save_path + output_vocab), 'w') as f:
            json.dump(word_freq, f)


def building_vocab(sequences, min_token_count=1):
    token_to_count = {}

    for i in sequences:
        for seqs in i[1]:
            for token in seqs:
                if token not in token_to_count:
                    token_to_count[token] = 0
                token_to_count[token] += 1

    token_to_idx = {}
    for token, inx in SPECIAL_TOKENS.items():
        token_to_idx[token] = inx

    for token, count in sorted(token_to_count.items()):
        if token in token_to_idx.keys():
            continue
        if count >= min_token_count:
            token_to_idx[token] = len(token_to_idx)

    return token_to_idx


def tokenize(s, delim=' ', add_start_token=True, add_end_token=True,
             punt_to_keep=None, punt_to_remove=None):
    if punt_to_keep is not None:
        for p in punt_to_keep:
            s = s.replace(p, '%s%s' % (delim, p))

        if punt_to_remove is not None:
            for p in punt_to_remove:
                s = s.replace(p, '')

        tokens = s.split(delim)
        for q in tokens:
            if q == '':
                tokens.remove(q)
        if tokens[0] == '':
            tokens.remove(tokens[0])
        elif tokens[-1] == '':
            tokens.remove(tokens[-1])

        if add_start_token:
            tokens = ['<START>'] + tokens
        if add_end_token:
            tokens.append('<END>')
        return tokens


def encoding_token(seq_tokens, token2idx, allow_unknown=False):
    seq_idx = []
    for token in seq_tokens:
        if token not in token2idx:
            if allow_unknown:
                token = '<UNK>'
            else:
                raise KeyError(f'{token} is unknown token in vocab')
        seq_idx.append(token2idx[token])
    return seq_idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='LEVIR_CC', help="dataset name")
    parser.add_argument("--word_count_threshold", type=int, default=5, help="word count threshold")

    args = parser.parse_args()
    main(args)

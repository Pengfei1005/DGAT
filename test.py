import time
import os
from torch.utils import data
import argparse
import json
from utils.LEVIR_CC_dataset import LEVIR_CC_Dataset
from utils.Dubai_CC_dataset import Dubai_CC_Dataset
from utils.WHU_CDC_dataset import WHU_CDC_Dataset
from utils.utils import *
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
        word_vocab = json.load(f)
    
    # Load train checkpoint
    checkpoint = torch.load(args.checkpoint)

    extractor = checkpoint['extractor']
    encoder = checkpoint['encoder']
    generator = checkpoint['generator']

    # Print dataset and model info
    print("dataset: {}".format(args.data_name))
    print("extractor: {}".format(args.network))
    print("encoder: {}".format('DGA'))
    print("generator: {}".format('Transformer'))

    # Eval mode
    extractor.eval()
    extractor = extractor.cuda()
    encoder.eval()
    encoder = encoder.cuda()
    generator.eval()
    generator = generator.cuda()

    # Custom dataloaders
    if args.data_name == 'LEVIR_CC':
        # LEVIR:
        nochange_list = ["the scene is the same as before ", "there is no difference ",
                         "the two scenes seem identical ", "no change has occurred ",
                         "almost nothing has changed "]
        test_loader = data.DataLoader(
            LEVIR_CC_Dataset(data_path=args.data_folder, list_path=args.list_path, split='test', 
                             max_length=args.max_length, allow_unknown=args.allow_unknown, 
                             vocab_file=args.vocab_file, token_path=args.token_folder),
            batch_size=args.test_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    elif args.data_name == 'Dubai_CC':
        # Dubai:
        nochange_list = ["Nothing has changed ", "There is no difference ", "All remained the same "
                         "Everything remains the same ", "Nothing has changed in this area ",
                         "Nothing changed in this area ", "No changes in this area ", "No change was made ",
                         "There is no change to mention ", "No changes to mention ", "No changed to mention ",
                         "No difference in this area ", "No change to mention ", "No change was made ",
                         "No change occurred in this area ", "The area appears the same "]
        test_loader = data.DataLoader(
            Dubai_CC_Dataset(data_path=args.data_folder, list_path=args.list_path, split='test', 
                             max_length=args.max_length, allow_unknown=args.allow_unknown, 
                             vocab_file=args.vocab_file, token_path=args.token_folder),
            batch_size=args.test_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    elif args.data_name == 'WHU_CDC':
        # WHUCDC:
        nochange_list = ["the scene is the same as before ", "there is no difference ",
                         "the two scenes seem identical ", "no change has occurred ",
                         "almost nothing has changed "]
        test_loader = data.DataLoader(
            WHU_CDC_Dataset(data_path=args.data_folder, list_path=args.list_path, split='test', 
                            max_length=args.max_length, allow_unknown=args.allow_unknown, 
                            vocab_file=args.vocab_file, token_path=args.token_folder),
            batch_size=args.test_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    # Upsample for Dubai
    l_resize1 = torch.nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
    l_resize2 = torch.nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
    
    # Record epochs
    test_start_time = time.time()
    references = list()  
    hypotheses = list()  
    change_references = list()
    change_hypotheses = list()
    nochange_references = list()
    nochange_hypotheses = list()
    change_acc = 0
    nochange_acc = 0

    # Eval
    with torch.no_grad():
        for ind, (imgA, imgB, token_all, token_all_len, _, _, _) in tqdm(enumerate(test_loader)):

            imgA = imgA.cuda()
            imgB = imgB.cuda()
            if args.data_name == 'Dubai_CC':
                imgA = l_resize1(imgA)
                imgB = l_resize2(imgB)
            token_all = token_all.squeeze(0).cuda()
            '''
            imageA_resized = F.interpolate(imgA, size=(224, 224), mode='bilinear', align_corners=True)
            imageB_resized = F.interpolate(imgB, size=(224, 224), mode='bilinear', align_corners=True)
            '''
            # Instantiate model
            feat1, feat2 = extractor(imgA, imgB)
            feat1, feat2 = encoder(feat1, feat2)
            seq,_ = generator.sample(feat1, feat2)

            img_token = token_all.tolist()

            img_tokens = list(map(lambda c: [w for w in c if w not in {word_vocab['<START>'], word_vocab['<END>'],
                                                                       word_vocab['<NULL>']}], img_token))

            references.append(img_tokens)

            pred_seq = [w for w in seq if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
            hypotheses.append(pred_seq)
            assert len(references) == len(hypotheses)

            # Check whether change or not
            pred_caption = ""
            ref_caption = ""
            for i in pred_seq:
                pred_caption += (list(word_vocab.keys())[i]) + " "
            ref_caption = ""
            for i in img_tokens[0]:
                ref_caption += (list(word_vocab.keys())[i]) + " "
            ref_captions = ""
            for i in img_tokens:
                for j in i:
                    ref_captions += (list(word_vocab.keys())[j]) + " "
                ref_captions += ".    "

            if ref_caption in nochange_list:
                nochange_references.append(img_tokens)
                nochange_hypotheses.append(pred_seq)
                if pred_caption in nochange_list:
                    nochange_acc = nochange_acc + 1
            else:
                change_references.append(img_tokens)
                change_hypotheses.append(pred_seq)
                if pred_caption not in nochange_list:
                    change_acc = change_acc + 1

        test_time = time.time() - test_start_time

        # Calculate evaluation scores
        print('len(nochange_references):', len(nochange_references))
        print('len(change_references):', len(change_references))

        if len(nochange_references) > 0:
            print('nochange_metric:')
            nochange_metric = get_eval_score(nochange_references, nochange_hypotheses)
            Bleu_1 = nochange_metric['Bleu_1']
            Bleu_2 = nochange_metric['Bleu_2']
            Bleu_3 = nochange_metric['Bleu_3']
            Bleu_4 = nochange_metric['Bleu_4']
            Meteor = nochange_metric['METEOR']
            Rouge = nochange_metric['ROUGE_L']
            Cider = nochange_metric['CIDEr']
            print('BLEU-1: {0:.4f}\t' 'BLEU-2: {1:.4f}\t' 'BLEU-3: {2:.4f}\t'
                  'BLEU-4: {3:.4f}\t' 'Meteor: {4:.4f}\t' 'Rouge: {5:.4f}\t' 'Cider: {6:.4f}\t'
                  .format(Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider))
            print("nochange_acc:", nochange_acc / len(nochange_references))
        if len(change_references) > 0:
            print('change_metric:')
            change_metric = get_eval_score(change_references, change_hypotheses)
            Bleu_1 = change_metric['Bleu_1']
            Bleu_2 = change_metric['Bleu_2']
            Bleu_3 = change_metric['Bleu_3']
            Bleu_4 = change_metric['Bleu_4']
            Meteor = change_metric['METEOR']
            Rouge = change_metric['ROUGE_L']
            Cider = change_metric['CIDEr']
            print('BLEU-1: {0:.4f}\t' 'BLEU-2: {1:.4f}\t' 'BLEU-3: {2:.4f}\t'
                  'BLEU-4: {3:.4f}\t' 'Meteor: {4:.4f}\t' 'Rouge: {5:.4f}\t' 'Cider: {6:.4f}\t'
                  .format(Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider))
            print("change_acc:", change_acc / len(change_references))

        score_dict = get_eval_score(references, hypotheses)
        Bleu_1 = score_dict['Bleu_1']
        Bleu_2 = score_dict['Bleu_2']
        Bleu_3 = score_dict['Bleu_3']
        Bleu_4 = score_dict['Bleu_4']
        Meteor = score_dict['METEOR']
        Rouge = score_dict['ROUGE_L']
        Cider = score_dict['CIDEr']
        print('Testing:\n' 'Time: {0:.3f}\t' 'BLEU-1: {1:.4f}\t' 'BLEU-2: {2:.4f}\t' 'BLEU-3: {3:.4f}\t'
              'BLEU-4: {4:.4f}\t' 'Meteor: {5:.4f}\t' 'Rouge: {6:.4f}\t' 'Cider: {7:.4f}\t'
              .format(test_time, Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DGAT for RSICC')
    
    
    # LEVIR_CC
    parser.add_argument('--data_folder', default='./data/LEVIR_CC/images', help='data files path')
    parser.add_argument('--list_path', default='./data/LEVIR_CC/', help='list path')
    parser.add_argument('--token_folder', default='./data/LEVIR_CC/tokens/', help='token files path')
    parser.add_argument('--vocab_file', default='vocab', help='vocab file')
    parser.add_argument('--max_length', type=int, default=41, help='max length of each caption sentence')
    parser.add_argument('--allow_unknown', type=int, default=1, help='whether unknown tokens are allowed')
    parser.add_argument('--data_name', default="LEVIR_CC", help='base name shared by data files.')
    parser.add_argument('--checkpoint',default='./DGAT_levir.pth', help='path to checkpoint')
    
    '''
    #Dubai-CC
    parser.add_argument('--data_folder', default='./data/Dubai_CC/images', help='data files path')
    parser.add_argument('--list_path', default='./data/Dubai_CC/', help='list path')
    parser.add_argument('--token_folder', default='./data/Dubai_CC/tokens/',help='token files path')
    parser.add_argument('--vocab_file', default='vocab', help='vocab file')
    parser.add_argument('--max_length', type=int, default=27, help='max length of each caption sentence')
    parser.add_argument('--allow_unknown', type=int, default=1, help='whether unknown tokens are allowed')
    parser.add_argument('--data_name', default="Dubai_CC", help='base name shared by data files.')
    parser.add_argument('--checkpoint',default='./DGAT_dubai.pth', help='path to checkpoint')
    '''
    '''
    #WHU_CDC
    parser.add_argument("--data_name", type=str, default='WHU_CDC', help='data name')
    parser.add_argument("--data_folder", type=str, default='./data/WHU_CDC/images/', help='data files path')
    parser.add_argument("--list_path", type=str, default='./data/WHU_CDC/', help='list path')
    parser.add_argument("--token_folder", type=str, default='./data/WHU_CDC/tokens/', help='token files path')
    parser.add_argument("--max_length", type=int, default=26, help='max length of each caption sentence')
    parser.add_argument("--allow_unknown", type=int, default=1, help='whether unknown tokens are allowed')
    parser.add_argument('--vocab_file', default='vocab', help='vocab file')
    parser.add_argument('--checkpoint',default='./DGAT_whucdc.pth',help='path to checkpoint')
    '''

    parser.add_argument('--network', default='segformer-mit_b1', help='define the encoder to extract features:resnet101,vgg16')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id in the training.')
    parser.add_argument('--workers', type=int, default=8,
                        help='for data-loading; right now, only 0 works with h5pys in windows.')
    parser.add_argument('--encoder_dim', default=512,
                        help='the dimension of extracted features using different network:2048,512')
    parser.add_argument('--feat_size', default=16, help='define the output size of encoder to extract features')
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=2048)
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--test_batchsize', default=1, help='batch_size for validation')



    args = parser.parse_args()
    main(args)

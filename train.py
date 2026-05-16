import argparse
import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from tqdm import tqdm
from utils.LEVIR_CC_dataset import LEVIR_CC_Dataset
from utils.Dubai_CC_dataset import Dubai_CC_Dataset
from utils.WHU_CDC_dataset import WHU_CDC_Dataset
from torch.nn.utils.rnn import pack_padded_sequence
from utils.utils import get_eval_score, accuracy
from models.backbone import Extractor
from models.transformer_decoder_inception import FullTransformerDecoder
from models.dga_encoder_cross import DifferenceEncoder


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_num_threads(2)

def main(args):

    log_file = os.path.join(args.save_path,"train_log.txt")
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # save checkpoint
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    print(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))

    if args.data_name == 'LEVIR_CC':
        start_epoch = 0
        best_bleu4 = 0.5
    elif args.data_name == 'Dubai_CC':
        start_epoch = 0
        best_bleu4 = 0.35
    else:
        start_epoch = 0
        best_bleu4 = 0.6

    # Read Word Map
    with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
        word_map = json.load(f)

    # Extractor
    extractor = Extractor(args.cnn_net)
    extractor.fine_tuning(args.fine_tune)

    # Difference Encoder
    encoder = DifferenceEncoder(n_layers=args.n_layer, feature_size=[args.feat_size, args.feat_size, args.encoder_dim], hidden_dim=args.hidden_dim, sigma_init=args.sigma_init, SEsigma_init = args.SE_sigma_init)

    # Caption Decoder
    generator = FullTransformerDecoder(encoder_dim=args.encoder_dim, feature_dim=args.feature_dim, vocab_size=len(word_map),
                                  max_lengths=args.max_length, word_vocab=word_map, n_head=args.n_heads,
                                  n_layers=args.decoder_n_layers, dropout=args.dropout)
    
    # Calculate and print parameters of each part
    extractor_params = sum(p.numel() for p in extractor.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    generator_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    total_params = extractor_params + encoder_params + generator_params

    print(f"Extractor Parameters: {extractor_params}")
    print(f"Encoder Parameters: {encoder_params}")
    print(f"Generator Parameters: {generator_params}")
    print(f"Total Parameters: {total_params}")
    print("\n")

    parameter_file = os.path.join(args.save_path,"parameters.txt")
    if not os.path.exists(os.path.dirname(parameter_file)):
        os.makedirs(os.path.dirname(parameter_file))
                    
    with open(parameter_file, "a") as f:
        f.write(f"Extractor Parameters: {extractor_params}" + "\n")
        f.write(f"Encoder Parameters: {encoder_params}" + "\n")
        f.write(f"Generator Parameters: {generator_params}" + "\n")
        f.write(f"Total Parameters: {total_params}" + "\n")
        f.flush()

    extractor_optimizer = torch.optim.Adam(extractor.parameters(), lr=args.cnn_lr)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.encoder_lr)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.decoder_lr)

    extractor = extractor.cuda()
    encoder = encoder.cuda()
    generator = generator.cuda()

    print("dataset: {}".format(args.data_name))
    print("extractor: {}".format(args.cnn_net))
    print("encoder: {}".format('DGA'))
    print("generator: {}".format('Transformer'))

    # Loss Function
    criterion = nn.CrossEntropyLoss().cuda()

    # Custom DataLoad
    if args.data_name == 'LEVIR_CC':
        train_dataloader = data.DataLoader(
            LEVIR_CC_Dataset(args.data_path, args.list_path, split='train', 
                                                            max_length=args.max_length, allow_unknown=args.allow_unknown, 
                                                            vocab_file=args.vocab_file, token_path=args.token_folder),
                                           batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        print("train_dataloader length {}".format(len(train_dataloader)))
        valid_dataloader = data.DataLoader(
            LEVIR_CC_Dataset(args.data_path, args.list_path, split='val',
                                                            max_length=args.max_length, allow_unknown=args.allow_unknown, 
                                                            vocab_file=args.vocab_file, token_path=args.token_folder),
                                           batch_size=args.valid_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        print("val_dataloader length {}".format(len(valid_dataloader)))

    elif args.data_name == 'Dubai_CC':
        train_dataloader = data.DataLoader(
            Dubai_CC_Dataset(args.data_path, args.list_path, split='train',
                                                            max_length=args.max_length, allow_unknown=args.allow_unknown, 
                                                            vocab_file=args.vocab_file, token_path=args.token_folder),
                                           batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        print("train_dataloader length {}".format(len(train_dataloader)))
        valid_dataloader = data.DataLoader(
            Dubai_CC_Dataset(args.data_path, args.list_path, split='val',
                                                            max_length=args.max_length, allow_unknown=args.allow_unknown, 
                                                            vocab_file=args.vocab_file, token_path=args.token_folder),
                                           batch_size=args.valid_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        print("val_dataloader length {}".format(len(valid_dataloader)))

    elif args.data_name == 'WHU_CDC':
        train_dataloader = data.DataLoader(
            WHU_CDC_Dataset(args.data_path, args.list_path, split='train',
                                                            max_length=args.max_length, allow_unknown=args.allow_unknown, 
                                                            vocab_file=args.vocab_file, token_path=args.token_folder),
                                           batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers,
                                           pin_memory=True)
        print("train_dataloader length {}".format(len(train_dataloader)))
        valid_dataloader = data.DataLoader(
            WHU_CDC_Dataset(args.data_path, args.list_path, split='val',
                                                            max_length=args.max_length, allow_unknown=args.allow_unknown, 
                                                            vocab_file=args.vocab_file, token_path=args.token_folder),
                                           batch_size=args.valid_batch_size, shuffle=True, num_workers=args.num_workers,
                                           pin_memory=True)
        print("val_dataloader length {}".format(len(valid_dataloader)))

    extractor_lr_scheduler = torch.optim.lr_scheduler.StepLR(extractor_optimizer, step_size=1, gamma=0.95)
    encoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=1, gamma=0.95)
    generator_lr_scheduler = torch.optim.lr_scheduler.StepLR(generator_optimizer, step_size=1, gamma=0.95)

    l_resizeA = torch.nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
    l_resizeB = torch.nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)

    index_i = 0
    hist = np.zeros((args.num_epochs * len(train_dataloader), 3))

    # Start Training

    for epoch in range(start_epoch, args.num_epochs):
        for id, (imgA, imgB, _, _, token, token_len, _) in enumerate(train_dataloader):
            # Train mode
            start_time = time.time()
            extractor.train()
            encoder.train()
            generator.train()

            # Zero grad
            if extractor_optimizer is not None:
                extractor_optimizer.zero_grad()
            encoder_optimizer.zero_grad()
            generator_optimizer.zero_grad()

            imgA = imgA.cuda()
            imgB = imgB.cuda()
            if args.data_name == 'Dubai_CC':
                imgA = l_resizeA(imgA)
                imgB = l_resizeB(imgB)
            token = token.squeeze(1).cuda()
            token_len = token_len.cuda()
            
            # Instantiate model
            feat_A, feat_B = extractor(imgA, imgB)
            feat_A, feat_B = encoder(feat_A, feat_B)
            score, caps_sorted, decode_lengths, sort_ind = generator(feat_A, feat_B, token, token_len)

            targets = caps_sorted[:, 1:]
            scores = pack_padded_sequence(score, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss
            loss = criterion(scores, targets)

            # Backward pass
            loss.backward()

            # Clip gradients
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_value_(encoder.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_value_(generator.parameters(), args.grad_clip)
                if encoder_optimizer is not None:
                    torch.nn.utils.clip_grad_value_(extractor.parameters(), args.grad_clip)

            # Update weights
            generator_optimizer.step()
            encoder_optimizer.step()
            if extractor_optimizer is not None:
                extractor_optimizer.step()

            # Track metric
            hist[index_i, 0] = time.time() - start_time
            hist[index_i, 1] = loss.item()
            hist[index_i, 2] = accuracy(scores, targets, 5)
            index_i += 1

            # Print status
            with open(log_file, "a") as f:
                if index_i % args.print_freq == 0:
                    train_message = ('Epoch: [{0}][{1}/{2}]\t'
                        'Batch Time: {3:.3f}\t'
                        'Loss: {4:.4f}\t'
                        'Top-5 Accuracy: {5:.3f}'.format(epoch, index_i, args.num_epochs * len(train_dataloader),
                                                        np.mean(hist[index_i - args.print_freq:index_i - 1, 0]) *
                                                        args.print_freq,
                                                        np.mean(hist[index_i - args.print_freq:index_i - 1, 1]),
                                                        np.mean(hist[index_i - args.print_freq:index_i - 1, 2])))
                    print(train_message)
                
                    f.write(train_message + "\n")
                    f.flush()

        # Eval mode
        if extractor is not None:
            extractor.eval()
        encoder.eval()
        generator.eval()

        val_start_time = time.time()
        references = list()  
        hypotheses = list()  

        with torch.no_grad():
            for id, (imgA, imgB, token_all, token_len, _, _, _) in tqdm(enumerate(valid_dataloader)):
                
                imgA = imgA.cuda()
                imgB = imgB.cuda()
                if args.data_name == 'Dubai_CC':
                    imgA = l_resizeA(imgA)
                    imgB = l_resizeB(imgB)
                token_all = token_all.squeeze(0).cuda()

                # Instantiate model
                if extractor is not None:
                    feat_A, feat_B = extractor(imgA, imgB)
                feat_A, feat_B = encoder(feat_A, feat_B)

                sequence,_ = generator.sample(feat_A, feat_B)

                img_token = token_all.tolist()

                img_tokens = list(map(lambda c: [w for w in c if w not in {word_map['<START>'], word_map['<END>'],
                                                                           word_map['<NULL>']}], img_token))
                references.append(img_tokens)

                pred_sequence = [w for w in sequence if
                                 w not in {word_map['<START>'], word_map['<END>'], word_map['<NULL>']}]
                hypotheses.append(pred_sequence)
                assert len(references) == len(hypotheses)

                if id % args.print_freq == 0:
                    pred_caption = ""
                    ref_caption = ""
                    for i in pred_sequence:
                        pred_caption += (list(word_map.keys())[i]) + " "
                    ref_caption = ""
                    for i in img_tokens:
                        for j in i:
                            ref_caption += (list(word_map.keys())[j]) + " "
                        ref_caption += ".    "

            val_time = time.time() - val_start_time

            # Calculate Metrics
            score_dict = get_eval_score(references, hypotheses)
            Bleu_1 = score_dict['Bleu_1']
            Bleu_2 = score_dict['Bleu_2']
            Bleu_3 = score_dict['Bleu_3']
            Bleu_4 = score_dict['Bleu_4']
            Meteor = score_dict['METEOR']
            Rouge_L = score_dict['ROUGE_L']
            Cider = score_dict['CIDEr']
            
            val_message = ('Validation:\n' 'Time: {0:.3f}\t' 'BLEU-1: {1:.4f}\t' 'BLEU-2: {2:.4f}\t' 'BLEU-3: {3:.4f}\t'
                  'BLEU-4: {4:.4f}\t' 'Meteor: {5:.4f}\t' 'Rouge: {6:.4f}\t' 'Cider: {7:.4f}\t'
                  .format(val_time, Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge_L, Cider))
            print(val_message)
            with open(log_file, "a") as f:
                f.write(val_message + "\n")
                f.flush()

        # Adjust Learning Rate
        generator_lr_scheduler.step()
        print("generator_lr: ", generator_optimizer.param_groups[0]['lr'])
        encoder_lr_scheduler.step()
        print("encoder_lr: ", encoder_optimizer.param_groups[0]['lr'])
        if extractor_lr_scheduler is not None:
            extractor_lr_scheduler.step()
            print("extractor_lr: ", extractor_optimizer.param_groups[0]['lr'])
        average_metric = (Bleu_4 + Meteor + Rouge_L + Cider) / 4
        # Check whether to save best model based on the val performance
        if Bleu_4 > best_bleu4:
            best_bleu4 = Bleu_4
            print("Save Model")

            state = {
                'extractor': extractor,
                'encoder': encoder,
                'generator': generator,
            }

            model_name = (str(args.data_name) + '_BatchSize' + '_' + str(args.train_batch_size) + '_' +
                        str(args.cnn_net) + '_' + 'Bleu-4' + '_' + str(round(10000 * Bleu_4)) + '_' +
                        str(round(10000 * average_metric)) + '_' + str(round(10000 * Cider)) + '_' + str(args.n_layer) + '_' + '1024' + 'test' + '.pth')
            torch.save(state, os.path.join(args.save_path, model_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DGAT for RSICC')

    parser.add_argument("--gpu_id", type=int, default=0, help='gpu id of the devices')
    parser.add_argument("--fine_tune", type=bool, default=True, help='fine tune cnn')
    parser.add_argument("--n_layer", type=int, default=1, help='number of layers')
    parser.add_argument("--decoder_n_layers", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=512, help='dimension of model')
    parser.add_argument("--n_heads", type=int, default=8, help='number of heads')
    parser.add_argument("--vocab_file", type=str, default='vocab', help='vocab file')
    parser.add_argument("--cnn_net", default='segformer-mit_b1', help='extractor network')
    parser.add_argument("--encoder_dim", type=int, default=512, help='the dim of extracted features by diff nets')
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=2048)
    parser.add_argument("--feat_size", type=int, default=16)
    parser.add_argument('--sigma_init', type=int, default=1, help='sigma_init')
    parser.add_argument('--SE_sigma_init', type=int, default=1, help='SE_sigma_init')
    

    #LEVIR-CC
    parser.add_argument("--data_name", type=str, default='LEVIR_CC', help='data name')
    parser.add_argument("--data_path", type=str, default='./data/LEVIR_CC/images/', help='data files path')
    parser.add_argument("--list_path", type=str, default='./data/LEVIR_CC/', help='list path')
    parser.add_argument("--token_folder", type=str, default='./data/LEVIR_CC/tokens/', help='token files path')
    parser.add_argument("--max_length", type=int, default=41, help='max length of each caption sentence')
    parser.add_argument("--allow_unknown", type=int, default=1, help='whether unknown tokens are allowed')
    parser.add_argument("--train_batch_size", type=int, default=32, help='batch size of training')

    '''
    # Dubai_CC
    parser.add_argument("--data_name", type=str, default='Dubai_CC', help='data name')
    parser.add_argument("--data_path", type=str, default='./data/Dubai_CC/images/', help='data files path')
    parser.add_argument("--list_path", type=str, default='./data/Dubai_CC/', help='list path')
    parser.add_argument("--token_folder", type=str, default='./data/Dubai_CC/tokens/', help='token files path')
    parser.add_argument("--max_length", type=int, default=27, help='max length of each caption sentence')
    parser.add_argument("--allow_unknown", type=int, default=1, help='whether unknown tokens are allowed')
    parser.add_argument("--train_batch_size", type=int, default=8, help='batch size of training')
    '''
    '''
    # WHU_CDC
    parser.add_argument("--data_name", type=str, default='WHU_CDC', help='data name')
    parser.add_argument("--data_path", type=str, default='./data/WHU_CDC/images/', help='data files path')
    parser.add_argument("--list_path", type=str, default='./data/WHU_CDC/', help='list path')
    parser.add_argument("--token_folder", type=str, default='./data/WHU_CDC/tokens/', help='token files path')
    parser.add_argument("--max_length", type=int, default=26, help='max length of each caption sentence')
    parser.add_argument("--allow_unknown", type=int, default=1, help='whether unknown tokens are allowed')
    parser.add_argument("--train_batch_size", type=int, default=32, help='batch size of training')
    '''

    parser.add_argument("--valid_batch_size", type=int, default=1, help='batch size of validation')
    parser.add_argument("--num_workers", type=int, default=8, help='to accelerate data load')
    parser.add_argument("--cnn_lr", type=float, default=1e-4, help='cnn learning rate')
    parser.add_argument("--encoder_lr", type=float, default=1e-4, help='encoder learning rate')
    parser.add_argument("--decoder_lr", type=float, default=1e-4, help='decoder learning rate')
    parser.add_argument("--num_epochs", type=int, default=50, help='number of epochs')
    parser.add_argument("--grad_clip", default=None, help='clip gradients')
    parser.add_argument("--print_freq", type=int, default=100, help='print frequency')
    parser.add_argument("--dropout", type=float, default=0.1, help='dropout')
    
    args = parser.parse_args()
    args.save_path = f"./{args.data_name}_checkpoints/"
    main(args)

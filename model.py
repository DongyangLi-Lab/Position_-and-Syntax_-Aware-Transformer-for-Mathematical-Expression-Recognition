import torch
from Encoder import CvT
from Tools import get_accuracy
from GrammarDecoder import DecoderTransformer


class model(torch.nn.Module):
    def __init__(self, config, vocab=None, styles=None):
        super(model, self).__init__()
        self.config = config
        self.styles = styles
        if vocab:
            self.vocab = vocab
        self.encoder = CvT(s3_emb_dim=self.config['model']['encoder_emb'],
                           **self.config['model']['cvt']).to(self.config['device'])
        if self.config['model']['decoder'] == 'transformer':
            self.decoder = DecoderTransformer(num_decoder_layers=self.config['model']['transformer']['num_decoder_layers'],
                                              emb_size=self.config['model']['encoder_emb'],
                                              tgt_vocab_size=self.config['model']['vocab_size'],
                                              dim_feedforward=self.config['model']['transformer']['hidden_size'],
                                              n_head=self.config['model']['transformer']['n_head'],
                                              dropout=self.config['model']['transformer']['dropout'],
                                              softmax=self.config['model']['softmax'],
                                              vocab=self.vocab, max_len=self.config['model']['max_len']).to(
                self.config['device'])
        elif self.config['model']['decoder'] == 'linear':
            self.decoder = torch.nn.Sequential(torch.nn.Linear(self.config['model']['encoder_emb'], self.vocab.__len__()).to(self.config['device']), torch.nn.LogSoftmax(dim=-1))
            self.decoder.train_generator = False
        else:
            raise Exception(f"decoder {self.config['model']['decoder']} not available, please choose one of [transformer, linear]")
        if self.config['train']['criterion'] == 'CrossEntropyLoss':
            self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        if self.config['train']['predict_style']:
            self.style_classifier = torch.nn.Linear(self.config['model']['encoder_emb'], len(styles)).to(self.config['device'])
            self.style_softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, images, labels, epoch, return_attn: bool = False):
        # encoder
        self.encoder.requires_grad_(True)
        if epoch <= self.config['train']['freeze_feature_extractor']:
            if 'feature_extractor' in self.encoder._modules:
                self.encoder._modules['feature_extractor'].requires_grad_(False)
        if epoch <= self.config['train']['freeze_encoder']:
            self.encoder.requires_grad_(False)

        features = self.encoder(images).to(self.config['device'])  # (S x B x C)
    
        # decoder
        labels_tgt = labels.permute(1, 0)
    
        attn_info = None
        if self.config['model']['decoder'] == 'transformer':
            # NOTE: our new decoder supports return_attn
            if (not self.decoder.training) and return_attn:
                out, attn_info = self.decoder(features, labels, return_attn=True)
            else:
                out = self.decoder(features, labels)
    
            if self.decoder.training:
                total_loss = self.criterion(out.permute((0, 2, 1)), labels_tgt[1:])
                predictions = torch.argmax(out.permute(1, 0, 2), -1)
            else:
                total_loss = 0
                predictions = []
                for out_i in out:
                    predictions.append(torch.argmax(out_i, -1).squeeze(dim=1))
        else:
            out = self.decoder(features)
            total_loss = self.criterion(out.permute((1, 2, 0)), labels_tgt[1:])
            predictions = torch.argmax(out, -1)
    
        acc = get_accuracy(labels=labels, predictions=predictions, end_number=self.vocab.token2id['_END_'])
    
        if return_attn and (attn_info is not None):
            return predictions, total_loss, acc, attn_info
        return predictions, total_loss, acc

    def forward_self_supervised(self, images, labels, epoch):
        # first and second round
        # encoder
        self.encoder.requires_grad_(True)
        if epoch <= self.config['train']['freeze_feature_extractor']:
            self.encoder._modules['feature_extractor'].requires_grad_(False)
        if epoch <= self.config['train']['freeze_encoder']:
            self.encoder.requires_grad_(False)
        features = self.encoder(images).to(self.config['device'])  # (t x b x c)
        # decoder
        labels_tgt = labels.permute(1, 0)
        if self.config['model']['decoder'] == 'transformer':
            out = self.decoder(features, labels)
            if self.decoder.training:
                total_loss = self.criterion(out.permute((0, 2, 1)), labels_tgt[1:])
                predictions = torch.argmax(out.permute(1, 0, 2), -1)
            else:
                total_loss = 0
                predictions = []
                for out_i in out:
                    predictions.append(torch.argmax(out_i, -1).squeeze(dim=1))
        else:
            out = self.decoder(features)
            total_loss = self.criterion(out.permute((1, 2, 0)), labels_tgt[1:])
            predictions = torch.argmax(out, -1)
        return predictions, total_loss, get_accuracy(labels=labels, predictions=predictions, end_number=self.vocab.token2id['_END_'])
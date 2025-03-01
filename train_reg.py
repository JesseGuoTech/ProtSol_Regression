from Solubilitylib import *
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler 

bert_model_path = "./model"
checkpoint_path = "./checkpoint"

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

class kmersCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels, **kwargs):
        super(kmersCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        inputs = inputs.unsqueeze(2).float()
        embeddings = torch.cat((inputs, inputs), dim=2)
        embeddings = embeddings.permute(0, 2, 1)
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs], dim=1)
        return encoding

class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers, bidirectional=True)

    def forward(self, inputs):
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        outputs, _ = self.encoder(embeddings)
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        return encoding

class bert_reg(nn.Module):
    def __init__(self, cnn_net, rnn_net):
        super(bert_reg, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_path)
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.5)
        self.drop3 = nn.Dropout(p=0.5)
        self.drop4 = nn.Dropout(p=0.2)
        self.cnn = cnn_net
        self.rnn = rnn_net
        self.outbio = nn.Linear(123, 512)
        self.relu = nn.ReLU()
        self.out = nn.Linear((800 + 256 + 512), 1)  # 输出层改为1个神经元

    def forward(self, input_ids):
        bioinfo = input_ids.pop("bioinfo")
        outputs = self.bert(**input_ids)
        pooled_out = outputs.pooler_output
        output = self.drop1(pooled_out)
        cnnout = self.cnn(output)
        rnn_out = self.rnn(input_ids['input_ids'])
        out1 = self.out(torch.cat([self.drop2(cnnout), self.drop3(rnn_out), self.drop4(self.relu(self.outbio(bioinfo)))], dim=1))
        return out1

def get_rnn(vocab_size=20, embed_size=64, num_hiddens=64, num_layers=2):
    net = BiRNN(vocab_size, embed_size, num_hiddens, num_layers)
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.LSTM:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    net.apply(init_weights)
    return net

def evaluate_loss_gpu(net, data_iter, loss, device=None):
    net.eval()
    if not device:
        device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    with torch.no_grad():
        for data in data_iter:
            x_input_ids, x_token_type_ids, x_attention_mask = data['input_ids'], data['token_type_ids'], data['attention_mask']
            x_bioinfo = data['bioinfo']
            X = {}
            X['input_ids'] = x_input_ids.cuda()
            X['token_type_ids'] = x_token_type_ids.cuda()
            X['attention_mask'] = x_attention_mask.cuda()
            X['bioinfo'] = x_bioinfo.cuda()
            y = data['labels'].cuda()
            y_hat = net(X)
            l = loss(y_hat, y.float())
            metric.add(l.sum(), y.numel())
    return metric[0] / metric[1]

def assement_accuracy_gpu(net, data_iter, device=None):
    net.cuda()
    net.eval()
    if isinstance(net, nn.Module):
        if not device:
            device = next(iter(net.parameters())).device
    i = 1
    with torch.no_grad():
        for data in data_iter:
            x_input_ids, x_token_type_ids, x_attention_mask = data['input_ids'], data['token_type_ids'], data['attention_mask']
            x_bioinfo = data['bioinfo']
            X = {}
            X['input_ids'] = x_input_ids.cuda()
            X['token_type_ids'] = x_token_type_ids.cuda()
            X['attention_mask'] = x_attention_mask.cuda()
            X['bioinfo'] = x_bioinfo.cuda()
            if i == 1:
                y_pred = net(X)
                y_true = data['labels']
            else:
                y_pred = torch.cat((y_pred, net(X)), 0)
                y_true = torch.cat((y_true, data['labels']), 0)
            i = i + 1
    return y_pred, y_true

def assesment(net, epoch, val_iter, dataset_name, path, devices=try_all_gpus()):
    y_pred, y_true = assement_accuracy_gpu(net, val_iter, device=devices)
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"MSE: {mse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")

def train_batch(net, X, y, loss, trainer, devices, scaler):
    if isinstance(X, dict):
        x_input_ids, x_token_type_ids, x_attention_mask = X['input_ids'], X['token_type_ids'], X['attention_mask']
        x_bioinfo = X['bioinfo']
        X = {}
        X['input_ids'] = x_input_ids.cuda()
        X['token_type_ids'] = x_token_type_ids.cuda()
        X['attention_mask'] = x_attention_mask.cuda()
        X['bioinfo'] = x_bioinfo.cuda()
    else:
        X = X.to(devices[0])
    net.train()
    trainer.zero_grad()
    with autocast():
        pred = net(X)
        l = loss(pred, y.float())  # 确保标签是浮点数
    scaler.scale(l).backward()
    scaler.step(trainer)
    scaler.update()
    train_loss_sum = l.sum()
    return train_loss_sum

def main(net, train_iter, test_iter, val_iter, val1_iter, loss, trainer, num_epochs, start_epoch, devices, checkpoint_path, test_record_path, scaler):
    logger = log_init(log_path="./log/train.log")
    timer, num_batches = Timer(), len(train_iter)
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                        legend=['train loss', 'test loss'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    best_loss = float('inf')
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)
    for epoch in range(start_epoch, num_epochs):
        net.train()
        metric = Accumulator(2)
        for i, data in enumerate(train_iter):
            timer.start()
            labels = data.pop('labels').cuda()
            features = data
            l = train_batch(net, features, labels, loss, trainer, devices, scaler)
            metric.add(l, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[1], None))
        net.eval()
        train_loss = metric[0] / metric[1]
        test_loss = evaluate_loss_gpu(net, test_iter, loss)
        val_loss = evaluate_loss_gpu(net, val_iter, loss)
        val1_loss = evaluate_loss_gpu(net, val1_iter, loss)

        animator.add(epoch + 1, (None, test_loss))
        logger.info(f'epoch {epoch + 1}/{num_epochs}: train loss {train_loss:.3f}, test loss {test_loss:.3f}, val loss {val_loss:.3f}, val1 loss {val1_loss:.3f}, {metric[1]*num_epochs/timer.sum():.1f} examples/sec on {str(devices)}')

        if test_loss < best_loss:
            best_loss = test_loss
            logger.info(f'################ epoch {epoch+1} .ckpt saved ################')
            checkpoint = {
                "net": net.module.state_dict(),
                'optimizer': trainer.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, checkpoint_path + '/epoch_' + str(epoch+1) + '_bestmodel.pkl')
            assesment(net, epoch, val_iter, dataset_name="val", path=test_record_path, devices=devices)
            assesment(net, epoch, val1_iter, dataset_name="val1", path=test_record_path, devices=devices)

if __name__ == '__main__':
    set_seed(1995)
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    rnn_net = get_rnn(vocab_size=tokenizer.vocab_size, embed_size=64, num_hiddens=64, num_layers=2)
    embed_size, kernel_sizes, nums_channels = 1, [1, 2, 3, 4, 5, 6, 7, 8], [100, 100, 100, 100, 100, 100, 100, 100]
    cnn_net = kmersCNN(tokenizer.vocab_size, embed_size, kernel_sizes, nums_channels)
    print_box("bert model loading...")
    modeltest = bert_reg(cnn_net=cnn_net, rnn_net=rnn_net)
    print_box("bert model loading completed...")
    print_box("data loading...")

    train_path = "./data/train.fasta"
    test_path = "./data/test.fasta"
    val_path = "./data/nesg.fasta"
    val1_path = "./data/chang.fasta"

    train_dataset = SolubilityDatasetBio(train_path, tokenizer=tokenizer, max_length=1024)
    test_dataset = SolubilityDatasetBio(test_path, tokenizer=tokenizer, max_length=1024)
    val_dataset = SolubilityDatasetBio(val_path, tokenizer=tokenizer, max_length=1024)
    val1_dataset = SolubilityDatasetBio(val1_path, tokenizer=tokenizer, max_length=1024)

    batch_size = 4
    train_iter = DataLoader(train_dataset, batch_size, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size, shuffle=True)
    val_iter = DataLoader(val_dataset, batch_size, shuffle=True)
    val1_iter = DataLoader(val1_dataset, batch_size, shuffle=True)

    print_box("dataset loading completed!")
    print_box("TRIANING BEGIN!!!")

    scaler = GradScaler()  # 训练前实例化一个GradScaler对象

    num_epochs = 40
    net = modeltest.cuda()
    net.out.apply(init_weights)

    trainer = torch.optim.Adam([{"params": net.bert.parameters(), "lr": 1e-7},
                                {"params": net.cnn.parameters(), "lr": 1e-5},
                                {"params": net.rnn.parameters(), "lr": 1e-5}],
                               lr=1e-7, weight_decay=1e-3)
    net.train()
    devices = try_all_gpus()
    loss = nn.MSELoss().cuda()  # 使用均方误差损失

    RESUME = False
    start_epoch = 0

    if RESUME:
        path_checkpoint = checkpoint_path + "/bestmodel.pkl"
        checkpoint = torch.load(path_checkpoint)
        net.load_state_dict(checkpoint['net'])
        net.cuda()
        trainer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    test_record_path = "./log"
    main(net, train_iter, test_iter, val_iter, val1_iter, loss, trainer, num_epochs, start_epoch, devices, checkpoint_path=checkpoint_path, test_record_path=test_record_path, scaler=scaler)
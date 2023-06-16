##written by Qihao Zhu
import torch
from torch import optim
from Dataset import SumDataset,ChunkedRandomSampler,rs_collate_fn1
import os
import time
from tqdm import tqdm
from Model import *
import numpy as np
from copy import deepcopy
import pickle
import sys
import json
from transformers import AutoModel, AutoTokenizer

from torch import multiprocessing as mp
from accelerate import Accelerator

import argparse
import random
sys.setrecursionlimit(500000000)

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
args = dotdict({
    'NlLen':512,
    'batch_size':144,
    "seed":42,
    'mask_id':0,
    'bertnum':0,
    "gradient_accumulation_steps":20,
    "patience":15,
    "max_num_trials":5,
    "max_rel_pos":10,
    'par': True,
    'use_apex':False,
    'max_grad_norm':1.0,
    'use_torch_amp':False,
    'mix_precision':'fp16',
    'task':'django',
    'eval':True,
    "pretrain_name":"grammart5-small"
})

def save_model(model, dirs='checkpointSearch/', optimizer=None, amp=None):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    if optimizer is not None:
        checkpoint = {
            'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'amp':amp.state_dict()
        }
        torch.save(checkpoint, dirs + 'best_model.ckpt')
    else:
        torch.save(model.state_dict(), dirs + 'best_model.ckpt')


def load_model(model, dirs = 'checkpointSearch/'):
    assert os.path.exists(dirs + 'best_model.ckpt'), 'Weights for saved model not found'
    model.load_state_dict(torch.load(dirs + 'best_model.ckpt', map_location='cpu'))
use_cuda = True#torch.cuda.is_available()
from transformers import get_linear_schedule_with_warmup
def finetune():
    args.gradient_accumulation_steps = 1
    taskconfig = json.loads(open('processdata/data/%s/config.json'%args.task, 'r').read())
    args.NlLen = taskconfig["NlLen"]
    # Initialize accelerator
    args.pretrain_name = "Salesforce/codegen-350M-mono"

    accelerator = Accelerator(mixed_precision=taskconfig['precision'], log_with='wandb')    
    args.batch_size = taskconfig["batch_size"] * 1 * accelerator.num_processes
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_name)
    if accelerator.is_main_process:
        data = pickle.load(open("processdata/%strain.pkl"%args.task, "rb"))
        datalen = len(data) // accelerator.num_processes
        print(datalen)
        for i in range(accelerator.num_processes):
            pickle.dump(data[i * datalen : (i + 1) * datalen], open("fttrain%d.pkl"%(i), "wb"))
        data = pickle.load(open("processdata/%svalid.pkl"%args.task, "rb"))
        if args.task in ['test']:
            data = data[:5000]
        if len(data) % accelerator.num_processes != 0:
            datalen = len(data) // accelerator.num_processes + 1
        else:
            datalen = len(data) // accelerator.num_processes
        for i in range(accelerator.num_processes):
            pickle.dump(data[i * datalen : (i + 1) * datalen], open("ftvalid%d.pkl"%(i), "wb"))
    accelerator.wait_for_everyone()
    hps = {"num_iterations": 100, "learning_rate": taskconfig['lr'], 'bs':taskconfig["batch_size"]}

    accelerator.init_trackers(args.task, config=hps)
    torch.manual_seed(args.seed + accelerator.process_index)        
    np.random.seed(args.seed + accelerator.process_index)
    random.seed(args.seed + accelerator.process_index)
    train_set = SumDataset(args, None, "train", idx=accelerator.process_index, mode='finetune', tokenizer1=tokenizer)   
    device = accelerator.device
    totoalnumber = len(train_set) * accelerator.num_processes
    model = Decoder1(args)
    #model.encoder.model.resize_token_embeddings(args.rulenum)
    #model.tie_word_embeddings()
    #load_model(model, 'checkpointEpchLR99Iter30010/')    

    optimizer = optim.AdamW(model.parameters(), eps=1e-8, lr=taskconfig["lr"])
    from transformers import get_linear_schedule_with_warmup

    global_step = 0
    train_size = args.batch_size // accelerator.num_processes // args.gradient_accumulation_steps
    accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = train_size

    model, optimizer = accelerator.prepare(model, optimizer)    
    accelerator.register_for_checkpointing(model)
    accelerator.register_for_checkpointing(optimizer)
    #accelerator.register_for_checkpointing(scheduler)
    num_trial = patience = 0
    isBetter = False
    #optimizer = ScheduledOptim(optimizer, d_model=args.embedding_size, max_steps=50000)
    maxAcc= 0
    maxC = 0
    minloss = 1e10
    global_step = 0
    avgruntime = 0        
    dev_set = SumDataset(args, None, "valid", idx=accelerator.process_index, tokenizer1=tokenizer)    
    if accelerator.is_main_process:
        open('communicate.txt', 'w').write('0')

    if args.eval:
        load_model(model.module, 'checkModel%s/'%args.task)    
    for epoch in range(1000):
        j = 0
        sampler = ChunkedRandomSampler(train_set, train_size)
        data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=train_size, drop_last=True, num_workers=10,collate_fn=rs_collate_fn1, sampler=sampler, pin_memory=True)
        for dBatch in tqdm(data_loader):
            isBetter = False
            if j % 100 == 20:                    
                accelerator.wait_for_everyone()
                #tnum, belu = evalmodel(dev_set, model, device, accelerator, tokenizer)
                tnum, belu = evalmodelacc(dev_set, model, device, accelerator)
                if accelerator.is_main_process:

                    open('communicate.txt', 'w').write('0')
                    if args.eval:
                        print('current acc and num %f %f'%(belu, tnum))
                        exit(0)
                    accelerator.log({"dev_bleu": belu, "dev_num": tnum})
                    print('current acc and num %f %f'%(belu, tnum))
                    if maxC < tnum:
                        maxC = tnum
                        unwrapped_model = accelerator.unwrap_model(model)
                        save_model(unwrapped_model, 'checkModelNUM/')
                        if taskconfig["metric"] == "acc":
                            isBetter = True
                        #print('find better accuracy %f'%tnum)
                        #save_model(model)
                    if maxAcc < belu:
                        if taskconfig["metric"] == "bleu":
                            isBetter = True
                        maxAcc = belu
                        print('find better acc %f'%belu)
                    if isBetter:
                        patience = 0
                        print('save model to [%s]' % 'checkModel%s/'%args.task, file=sys.stderr)
                        #save_model(model.module, 'checkModel%d-%d/'%(epoch, j))
                        unwrapped_model = accelerator.unwrap_model(model)
                        save_model(unwrapped_model, 'checkModel%s/'%args.task)
                        #accelerator.save_state('checkpoint/')
                        os.system('cp out.txt out1.txt')

                    elif patience < args.patience:
                        patience += 1
                        print('hit patience %d' % patience, file=sys.stderr)
                        if patience == args.patience:
                            num_trial += 1
                            print('hit #%d trial' % num_trial, file=sys.stderr)
                            if num_trial == args.max_num_trials:
                                print('early stop!', file=sys.stderr)
                                exit(0)
                            #lr = optimizer.param_groups[0]['lr'] * 0.5
                            open('communicate.txt', 'w').write('1')
                            #accelerator.load_state('checkpoint/')
                            patience = 0
                    else:
                        patience += 1
                accelerator.wait_for_everyone()                    
                reloads = open('communicate.txt', 'r').read()
                if reloads == '1':
                    load_model(model.module, 'checkModel%s/'%args.task)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = 0.5 * param_group['lr']                    
                    print('reload')
                accelerator.wait_for_everyone()
            model.train()
            for x in ((dBatch)):
                dBatch[x] = dBatch[x].to(device)
            starttime = time.time()
            loss, _ = model(dBatch['input'], dBatch['mask'])
            avgruntime += time.time() - starttime
            resmask = torch.eq(dBatch['mask'][:, :-1], 2)
            loss = torch.sum(loss) / torch.sum(resmask)
            if loss.sum().item() == np.inf:
                print('inf')
                exit(0)
            if loss.item() == np.inf:
                print(j)
                assert(0)
            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps
            if j % args.gradient_accumulation_steps == 0:
                accelerator.backward(loss)
            else:
                with accelerator.no_sync(model):
                    accelerator.backward(loss)
            if j % args.gradient_accumulation_steps == 0:
                optimizer.step()#_and_update_lr()
                optimizer.zero_grad()
                #scheduler.step()
            accelerator.log({"loss": loss.item()})
            #wandb.log({"loss": loss.item()})
            j += 1
            global_step += 1
        accelerator.log({"runtime": avgruntime / j})
            #display("metrics")
            #display("assets")
@torch.no_grad()
def evalmodel(dev_set, model, device, accelerator, tokenizer):
    data_loader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=1, drop_last=False, num_workers=2,collate_fn=rs_collate_fn1, shuffle=False, pin_memory=True)
    beamsize = 1
    model.eval()
    f = open("outval%d.txt"%int(accelerator.process_index), "w")
    for dBatch in tqdm(data_loader):
        dBatch['testi'] = dBatch['testi'].to(device)
        model.model.pad_token_id = None
        ans = model.model.generate(dBatch['testi'], max_length=1024, num_beams=beamsize, pad_token_id=tokenizer.eos_token_id)
        for i in range(len(ans)):
            f.write(tokenizer.decode(ans[i].tolist()[dBatch['testi'][0].size(0):], skip_special_tokens=True) + '\n')
            f.flush()
    f.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        for i in range(accelerator.num_processes):
            os.system('cat outval%d.txt >> outval.txt'%i)
        tnum, codebelu = calc_code_bleu.get_codebleu("processdata/groundvalid%s.txt"%args.task, "outval.txt", testlang, benchmark=args.task)
        return tnum, codebelu
    else:
        return 0, 0
@torch.no_grad()
def evalmodelacc(dev_set, model, device, accelerator):
    data_loader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=10, drop_last=False, num_workers=2,collate_fn=rs_collate_fn1, shuffle=False, pin_memory=True)
    accs = []
    tnums = []
    model.eval()
    deta = []
    for i, dbatch in enumerate(data_loader):
        dbatch['input'] = dbatch['input'].to(device)
        dbatch['mask'] = dbatch['mask'].to(device)
        loss, pred = model(dbatch['input'], dbatch['mask'])
        
        currentmask = dbatch['mask'][:, :-1]
        resmask = torch.eq(currentmask, 2)
        pre = pred.argmax(dim=-1)        
        ans = dbatch['input'][:, 1:]
        accnum = (torch.eq(ans, pre) * resmask).sum(dim=-1)
        acc = accnum.float() / resmask.sum(dim=-1).float()
        accs.append(acc.mean().item())
        tnum = torch.eq(accnum, resmask.sum(dim=-1))
        deta.extend(tnum.tolist())
        tnums.append(tnum.sum().item())
        
    tnum = np.sum(tnums)
    acc = np.mean(accs)
    open("resdetail%d.txt"%int(accelerator.process_index), "w").write(str(deta))
    open("res%d.txt"%int(accelerator.process_index), "w").write(str(acc) + "\t" + str(tnum))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accs = []
        tnums = []
        for i in range(accelerator.num_processes):
            acc, tnum = open("res%d.txt"%i).read().split()
            accs.append(float(acc))
            tnums.append(int(tnum))
        return np.sum(tnums), np.mean(accs)
    else:
        return 0, 0
@torch.no_grad()
def evalmodelnl(dev_set, model, device, accelerator):
    data_loader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=25, drop_last=False, num_workers=2,collate_fn=rs_collate_fn1, shuffle=False, pin_memory=True)
    from beamsearch import BeamSearch
    beamsize = 1
    beam = BeamSearch(beamsize, dev_set.ruledict, 0)
    model.eval()
    f = open("outval%d.txt"%int(accelerator.process_index), "w")
    for dBatch in tqdm(data_loader):
        dBatch['nl'] = dBatch['nl'].to(device).repeat_interleave(beamsize, dim=0)
        ans = beam.search(dBatch['nl'], model, max_len=args.CodeLen, vocabsize=args.rulenum, mode='nl')        
        for i in range(len(ans)):
            beamone = ans[i].set[0]
            root = beam.convertrulelist2tree(beamone.state, mode='nl')
            f.write(root)
            f.write("\n")
            f.flush()
    f.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        preds = []
        for i in range(accelerator.num_processes):
            f = open("outval%d.txt"%i, "r")
            for line in f:
                preds.append(str(len(preds)) + '\t' + line.strip())
            f.close()
        f = open("out.txt", "w")
        lines = open("processdata/groundvalid%s.txt"%args.task, "r").readlines()
        for i in range(len(preds)):
            f.write(str(i) + '\t' + lines[i].strip() + '\n')
        from bleunl import calbleu
        bleu = calbleu('out.txt', preds)
        return 0, bleu
    else:
        return 0, 0
if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='pretrain', type=str, required=True)
    argc = parser.parse_args()
    args.task = argc.dataset
    if args.task in ['django', 'concode', 'codetrans', 'repair', 'assert', 'conala', 'hs', 'test', 'repairme', 'transj2c', 'transc2j', 'commentjava', 'commentpython', 'mbpp']:
        args.eval = False
        finetune()
    if args.task in ['pretrain']:
        pretrain()
    if args.task in ['pretrain2']:
        pretrain2()
    if args.task in ['fill']:
        testfill()
    if args.task in ['searchadv', 'searchcos']:
        #args.eval = False
        from runsearch import finetune_search
        finetune_search(args)
    #pretrain2()
    '''if sys.argv[1] == "train": 
        train()
    elif sys.argv[1] == "eval": 
        eval()
    elif sys.argv[1] == "test": 
        profile = line_profiler.LineProfiler()
        profile.enable()
        test()
        profile.disable()
        profile.print_stats(sys.stdout)
    else:
        testone()'''
     #test()

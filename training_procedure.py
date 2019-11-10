from datetime import datetime
from time import time
from tqdm import tqdm
from glob import glob
import numpy as np
import torch as th

from networks.MSGNet import MSGNet


def dt():
    return datetime.now().strftime('%H:%M:%S')

def compute_loss(sample, model, criterion):
    ''' GPU running is assumed by default'''
    training_type = model.training_type

    if training_type == 'ordinary':
        Dx8_hf = sample['Dx8_hf'].cuda()
        Dx1_hf = sample['Dx1_hf'].cuda()
        Y_hf   = sample['Y_hf'].cuda()
        prediction = model(Y_hf, Dx1_hf)
        loss = criterion(prediction, Dx8_hf) 

    elif training_type == 'multiloss':
        Dx8_hf = sample['Dx8_hf'].cuda()
        Dx4_hf = sample['Dx4_hf'].cuda()
        Dx2_hf = sample['Dx2_hf'].cuda()
        Dx1_hf = sample['Dx1_hf'].cuda()
        Y_hf   = sample['Y_hf'].cuda()
        prediction = model(Y_hf, Dx1_hf)
        loss = criterion(prediction, Dx8_hf)
        for target, out in zip([Dx2_hf, Dx4_hf], model.each_scale_outs):
            loss += criterion(out, target)

    elif training_type == 'full_freq':
        Dx8_hf = sample['Dx8_hf'].cuda()
        Dx8_lf = sample['Dx8_lf'].cuda()
        Dx1_hf = sample['Dx1_hf'].cuda()
        Y_hf   = sample['Y_hf'].cuda() 
        prediction = model(Y_hf, Dx1_hf) + Dx8_lf
        clamper = th.nn.Hardtanh(model.depth_min, model.depth_max)
        prediction = clamper(prediction)
        loss = criterion(prediction, Dx8_hf+Dx8_lf)

    return loss

def train(model, optimizer, criterion, data_loader, save_path):
    ''' Train step. '''
    print('\n Training...')
    train_losses = list(np.load(save_path+'train_losses.npy'))

    for sample in tqdm(data_loader):
        
        optimizer.zero_grad()
        loss = compute_loss(sample, model, criterion)
        cur_loss = loss.data.cpu().numpy().item()
        train_losses.append(cur_loss)
        loss.backward()
        optimizer.step()

        np.save(save_path+'train_losses.npy', train_losses)


def val(model, criterion, data_loader, save_path):
    ''' Validation step. '''
    print('\n Validation...')
    val_losses = list(np.load(save_path+'val_losses.npy'))
    val_loss_per_epoch, num_losses_epoch = 0, 0 # for scheduler step

    for sample in tqdm(data_loader):
    # during validation only 'ordinary' mode is used

        Dx8_hf = sample['Dx8_hf'].cuda()
        Dx1_hf = sample['Dx1_hf'].cuda()
        Y_hf   = sample['Y_hf'].cuda()

        with th.no_grad(): prediction = model(Y_hf, Dx1_hf)
        loss = criterion(prediction, Dx8_hf)

        cur_loss = loss.data.cpu().numpy().item()
        val_losses.append(cur_loss)

        val_loss_per_epoch += cur_loss
        num_losses_epoch += 1

        np.save(save_path+'val_losses.npy', val_losses)

    return val_loss_per_epoch / num_losses_epoch

def save_model(epoch, model, optimizer, scheduler, save_path, save_each_epoch=50):
    ''' Saves the model state correctly. '''
    if epoch % save_each_epoch != 0 and epoch != 1:
        return

    state = {'model_state_dict':model.state_dict(),\
             'optimizer_state_dict':optimizer.state_dict(),\
             'scheduler_state_dict':scheduler.state_dict(),\
             'rng_state':th.get_rng_state(),\
             'rng_states_cuda':th.cuda.get_rng_state_all()}   

    np.save(save_path+'last_saved_epoch.npy', epoch)
    path2file = save_path+'state_'+'{:04d}'.format(epoch)+'.pth'
    th.save(state, path2file)

    print("\n ===>", dt(), "epoch={}. ".format(epoch))
    print("Model is saved to {}".format(save_path))

def load_model(epoch, model, optimizer, scheduler, save_path):
    ''' Loads the model state from the "~/saved_models/...". '''
    path2file = save_path+'state_'+'{:04d}'.format(epoch)+'.pth'

    state = th.load(path2file,map_location=lambda storage,loc:storage.cuda())
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])  
    scheduler.load_state_dict(state['scheduler_state_dict'])  
    th.set_rng_state(state['rng_state'].cpu()) 

    return model, optimizer, scheduler

def compute_training_time(start, time_prev):
    training_time = time() - start + time_prev
    
    days = int(training_time // 60 // 60 // 24)
    training_time -= days*60*60*24
    hours = int(training_time // 60 // 60)
    training_time -= hours*60*60
    minutes = int(training_time // 60)
    training_time -= minutes*60
    seconds = int(training_time)
    out = ''
    for time_slot, time_spell in zip([days, hours, minutes, seconds],\
                                     ['d', 'h', 'm', 's']):
        if time_slot != 0:
            out += str(time_slot)+' '+time_spell+' '
    return out

def training_procedure(start_epoch, num_epochs, model, optimizer, scheduler, \
                        criterion, train_loader, val_loader, 
                        save_path, save_each_epoch, no_val):

    # Procedure of training and validating  
    training_time = np.load(save_path+'training_time.npy').item()
    start = time()
    for epoch in range(start_epoch, num_epochs+1):
        # train
        train(model, optimizer, criterion, train_loader, save_path)
        print('\n', dt(), '; Training time: ', \
            compute_training_time(start, training_time), '\n')
        print('epoch={}/{}. Train is done.'.format(epoch, num_epochs))  

        if not no_val:
            # val
            val_loss = val(model, criterion, val_loader, save_path)
            # scheduler.step(val_loss)

            print('\n', dt())
            print('epoch={}/{}. Validation is done.'.format(epoch, num_epochs))  

        # save model
        save_model(epoch, model, optimizer, scheduler, \
            save_path, save_each_epoch)
        np.save(save_path+'training_time.npy', time()-start+training_time)
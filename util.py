import numpy as np
import torch
import copy

def get_initial_guess(coeff_matrix, noises, grad_requires=True):
    coef = copy.copy(coeff_matrix)
    coef = coef[None, None,:,:]
    init_guess=torch.linalg.lstsq(coef, noises,rcond=1e-5).solution
    #rcond 1e-5*largest eigenvalue
    # make sure no regularization
    ### .lstsq try it.
    if grad_requires:
        init_guess.requires_grad=True
    return  init_guess

def samples_generate(ground_truth, initial_guess):
    eps = torch.rand(1)
    samples = eps*ground_truth + (1-eps)*initial_guess
    if not samples.requires_grad:
        samples.requires_grad=True 
    return samples

def get_batch_instance(dataloader):
    instance_batch={}
    for images, labels, noises in dataloader:  
        instance_batch['image']=images
        instance_batch['label']=labels
        instance_batch['noise']=noises
        break
    
    return instance_batch

def get_instance(dataloader):
    instance={}
    for images, labels, noises in dataloader:  
        instance['image']=images[0]
        instance['label']=labels[0]
        instance['noise']=noises[0]
        break
    
    return instance


def Frobenius_distance(batch_instance, recovered_images, batch_size): 
    loss=0
    for i in range(batch_size):
        gr = batch_instance['images'][i][None,]
        recovered_image = recovered_images[i]
        loss+=torch.sqrt(torch.square(gr - recovered_image).sum(dim=(1,2,3)))
    return loss/batch_size

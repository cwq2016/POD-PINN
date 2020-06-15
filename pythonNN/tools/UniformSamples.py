# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 21:57:50 2020

@author: 56425
"""
import numpy as np

def UniformSamples(design_space,Nlevel,file):
    lb = design_space[0:1,:]
    ub = design_space[1:2,:]
    nvars = design_space.shape[1]
    xp = np.linspace(-1,1,Nlevel)
    xgrids = np.meshgrid(*( xp, )*nvars)
    xgrids = np.stack(xgrids, axis = nvars).reshape((-1,nvars))
    Samples= (ub+lb)/2 +xgrids*(ub-lb)/2
    
    if file:
        with open(file, 'w') as f:
            f.write("#the number of samples\n")
            f.write('%d\n'%Samples.shape[0]);
            formatstr = '%14.7e\t'*nvars + "\n"
            f.write('#lower bound\n');
            f.write(formatstr%tuple(lb[0,:]));
            f.write('#upper bound\n');
            f.write(formatstr%tuple(ub[0,:]));
            formatstr = '%d\t'+'%14.7e\t'*nvars+"\n"
            for i in range(Samples.shape[0]):
                f.write(formatstr%(i+1,*tuple(Samples[i,:]),));
    return Samples

if __name__ == "__main__":
    # lidDriven: Re, Theta
#    design_space =np.array([[100, 60],[200,120]])
#    Nlevel = 11
#    file ="./LidDrivenValidation.txt"
#    Samples = UniformSamples(design_space,Nlevel,file)
#    import matplotlib.pyplot as plt
#    plt.plot(Samples[:,0], Samples[:,1],'*')
    
    # Natural Convection: Ra, Pr, Theta
    design_space =np.array([[5E5, 0.7, 60],[1+5E5,0.71,90]])
    design_space =np.array([[1E5, 0.7, 60],[2E5,0.71,61]])
    design_space =np.array([[1E5, 0.6, 60],[3E5,0.8,90]])  
    design_space =np.array([[1E4, 0.6,  0],[1E5,0.8,90]])
    design_space =np.array([[1E4, 0.6, 60],[1E5,0.8,90]])  
    design_space =np.array([[1E4, 0.6, 60],[3E5,0.8,90]])
    design_space =np.array([[1E4, 0.6, 45],[1E5,0.8,90]])  
    Nlevel = 6
    file ="./NaturalConvectionValidation7.txt"
    Samples = UniformSamples(design_space,Nlevel,file)
    
    
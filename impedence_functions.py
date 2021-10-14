# ------------------------------- #
# ------- Package imports ------- #
# ------------------------------- #
 
import numpy as np
import sys
import time
import os
import html
import cProfile 
import re
from glob import glob
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('TkAgg') # this is for the backend on MAC: https://markhneedham.com/blog/2018/05/04/python-runtime-error-osx-matplotlib-not-installed-as-framework-mac/
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import scipy
from scipy import optimize
from scipy.optimize import curve_fit
from cxroots import Circle
import imageio
import mpmath as mp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


def paper_results():

    plot_dos()
    plot_q2b()
    plot_supp()

def dos_plotting_wrapper():

    alpha_arr=[0.01,0.05,0.1,0.4,0.9]
    psi_arr=[0.05,0.1,0.5,0.9,1.2,1.4]
    
    colors=["#111d5e","#c70039","#f37121","#ffbd69","#07031a","#438a5e"]
    eps_values=construct_eps_arr()
    
    for alpha in alpha_arr:
        gamma=(np.abs(1-alpha))/(1+alpha)
        print(" --- Doing gamma ="+ str(np.round(gamma,2))+ " --- ")    
        psi_fig,psi_ax=plt.subplots()
        for (psi,psi_color) in zip(psi_arr,colors):
            psi_alpha=psi/(1+alpha)**2            
            print("    ---> Doing psi = ", psi, " <--- ")
            print("    ---> Doing psi_alpha = ", psi_alpha, " <--- ")            
            print("        --->  GF creation <---  ")            
            gf,gf_r,gf_i=create_gf(eps_values,gamma,psi_alpha)
            print("        --->  DOS plotting <---  ")            
            dos_plotting(eps_values,gf_i,psi_ax,psi_color,psi)
        plotting_spec(psi_fig,psi_ax,gamma)

def impedence_setup():
    alpha_arr=[0.1]#,0.05,0.1,0.25,0.6,0.8]    
    psi=0.1
    eps_values=construct_eps_arr()
    alpha_color="#111d5e"
    colors=["#111d5e","#c70039","#f37121","#ffbd69","#07031a","#438a5e"]
    gamma_arr,psi_alpha_arr=[],[]
    gf_arr,gf_r_arr,gf_i_arr=[],[],[]
    v_r_arr,v_i_arr=[],[]
    h_r_arr,h_i_arr=[],[]
    for alpha in alpha_arr:
        gamma=(np.abs(1-alpha))/(1+alpha)
        psi_alpha=psi/(1+alpha)**2
        print("gamma = " +str(np.round(gamma,2)))
        gamma_arr.append(gamma), psi_alpha_arr.append(psi_alpha)   
        gf,ff,v=create_gf(eps_values,gamma,psi_alpha)
        gf_r,gf_i=decompose_gf(gf)
        v_r,v_i=decompose_gf(v)
        gf_arr.append(gf), gf_r_arr.append(gf_r), gf_i_arr.append(gf_r)
        v_r_arr.append(v_r), v_i_arr.append(v_i)
        hf=create_h(eps_values,gf,ff)
        h_r,h_i=decompose_gf(hf)        
        h_r_arr.append(h_r), h_i_arr.append(h_i)        
#        plt.plot(eps_values,h_i,marker='o')
    plot_v_imag(eps_values, v_i_arr, gamma_arr,colors,psi)
    plot_v_real(eps_values, v_r_arr, gamma_arr,colors,psi)
    dos_plotting_singular_real(eps_values,gf_r,alpha_color,gamma,psi)
    dos_plotting_singular_imag(eps_values,gf_i,alpha_color,gamma,psi)
    #    plot_h_imag()
    #    plot_h_real()
    #    dos_plotting(eps_values,gf_i,psi_ax,psi_color,psi)

    
def dos_animiation_wrapper():
    alpha_arr=[0.01,0.05,0.1,0.25,0.6,0.8]
    
    psi=0.3
    #    psi_arr=[0.05,0.1,0.5,0.9,1.2,1.4]
    colors=["#111d5e","#c70039","#f37121","#ffbd69","#07031a","#438a5e"]
    eps_values=construct_eps_arr()
    alpha_color="#111d5e"
    gamma_arr,psi_alpha_arr=[],[]
    gf_arr,gf_r_arr,gf_i_arr=[],[],[]
    v_r_arr,v_i_arr=[],[]
    for alpha in alpha_arr:
        gamma=(np.abs(1-alpha))/(1+alpha)
        psi_alpha=psi/(1+alpha)**2
        print("gamma = " +str(np.round(gamma,2)))
        gamma_arr.append(gamma)
        psi_alpha_arr.append(psi_alpha)   
        gf,f,v=create_gf(eps_values,gamma,psi_alpha)
        gf_r,gf_i=decompose_gf(gf)
        v_r,v_i=decompose_gf(v)
        gf_arr.append(gf)
        gf_r_arr.append(gf_r)
        gf_i_arr.append(gf_i)
        v_r_arr.append(v_r)
        v_i_arr.append(v_i)

    plot_v_real(eps_values,v_r_arr,gamma_arr,colors,psi)
    plot_v_imag(eps_values,v_i_arr,gamma_arr,colors,psi)
    
    plot_name="test.gif"
    kwargs_write = {'fps':1.0, 'quantizer':'nq'}
#    imageio.mimsave(plot_name,
#                    [dos_plotting_anim(eps_values,gf_r,alpha_color,gamma_val) for gamma_val,gf_r in zip(gamma_arr,gf_r_arr)],
#                    fps=1)
    imageio.mimsave(plot_name,
                    [dos_plotting_anim(eps_values,gf_i,alpha_color,gamma_val) for gamma_val,gf_i in zip(gamma_arr,gf_i_arr)],
                    fps=1)    
    print("Saved " + plot_name)

        
def construct_eps_arr():

    eps_step=0.01
    
    eps_lower_initial=0.0
    eps_lower_boundary=0.1
    eps_middle_boundary=1.5
    eps_upper_boundary=2.0
    
    eps_lower_region_step=eps_step
    eps_middle_region_step=eps_step
    eps_upper_region_step=eps_step

    eps_upper=np.arange(eps_middle_boundary,eps_upper_boundary,eps_upper_region_step)
    eps_middle=np.arange(eps_lower_boundary,eps_middle_boundary,eps_middle_region_step)
    eps_lower=np.arange(eps_lower_initial,eps_lower_boundary,eps_lower_region_step)
        
    eps_values=np.concatenate((eps_lower,eps_middle,eps_upper))

    return eps_values

def construct_k_arr():

    kmin=0.01
    kmax=0.02
    kstep=0.1
    karr=np.arange(kmin,kmax,kstep)
    
    return karr


def create_h(eps_arr,g_arr,f_arr):
    h_arr=[]
    for eps,g,f in zip(eps_arr, g_arr, f_arr): 
        h = 0.5*(eps/g + 1.0/f) # need to decide of value for gap
        h_arr.append(h)
    return h_arr

def create_gf(eps_arr,gamma,psi_alpha):
    
    delta=0.02
    v_arr=[]
    v_arr_real=[]
    v_arr_imag=[]
    gf_arr=[]
    f_arr=[]
    for eps in eps_arr:
#        iv=eps+delta + 1.0*mp.j # initial value: shift to complex for safety
        iv=eps+delta + psi_alpha*mp.j # initial value: shift to complex for safety: normalise by psi_alpha
        gf=mp.findroot(lambda x: x*(1 - psi_alpha*(mp.sqrt(1-x**2)/(gamma**2-x**2))) - eps,
                       iv,
                       solver='muller')
        v_arr.append(gf)
        v_arr_real.append(gf.real)
        v_arr_imag.append(gf.imag)
        gf_arr.append(gf/mp.sqrt(gf**2-1))
        f_arr.append(1/mp.sqrt(gf**2-1))

    # plt.plot(eps_arr, v_arr_real)
    # plt.title(r"real $gamma =$  "+str(np.round(gamma,2)))
    # plt.show()

    return gf_arr, f_arr, v_arr

def decompose_gf(gf_arr):
    gf_real_arr=[]
    gf_imag_arr=[]
    for gf_obj in gf_arr:
        gf_real_arr.append(float(abs(gf_obj.real)))
        gf_imag_arr.append(float(abs(gf_obj.imag)))
    return gf_real_arr,gf_imag_arr

def dos_plotting(eps_arr,gf_arr,ax,plot_color,psi_val):

    arr_max=np.amax(np.asarray(gf_arr)) # choose correct root
    eps_boundary=np.where(eps_arr>1.0)[0][0] # where eps>0 starts
    lw=2
    match_impurity_band_with_continuum=True
    
    if match_impurity_band_with_continuum: 
        ax.plot(eps_arr[:eps_boundary],np.asarray(gf_arr[:eps_boundary])/arr_max,
                color=plot_color,
                linewidth=lw,
                label=r"$\psi=$"+str(psi_val))
    else:
        ax.plot(eps_arr[:eps_boundary],np.asarray(gf_arr[:eps_boundary]),
                color=plot_color,
                linewidth=lw,
                label=r"$\psi=$"+str(psi_val))
            
    ax.plot(eps_arr[eps_boundary:],np.asarray(gf_arr[eps_boundary:])/arr_max,
            color=plot_color,
            linewidth=lw)



def dos_plotting_anim(eps_arr,gf_arr,plot_color,gamma_val):
    
    psi_fig,psi_ax=plt.subplots()
    arr_max=np.amax(np.asarray(gf_arr))               
    eps_boundary=np.where(eps_arr>1.0)[0][0] # where eps>0 starts
    lw=2
    match_impurity_band_with_continuum=True
    
    if match_impurity_band_with_continuum: 
        psi_ax.plot(eps_arr[:eps_boundary],np.asarray(gf_arr[:eps_boundary])/arr_max,
                    color=plot_color,
                    linewidth=lw,
                    label=r"$\gamma= $"+str(np.round(gamma_val,2)))
    else:
        psi_ax.plot(eps_arr[:eps_boundary],np.asarray(gf_arr[:eps_boundary]),
                    color=plot_color,
                    linewidth=lw,
                    label=r"$\gamma= $"+str(np.round(gamma_val,2)))
            
    psi_ax.plot(eps_arr[eps_boundary:],np.asarray(gf_arr[eps_boundary:])/arr_max,
                color=plot_color,
                linewidth=lw)

    plotting_spec_anim(psi_fig,psi_ax)

    # Used to return the plot as an image rray
    psi_fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(psi_fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(psi_fig.canvas.get_width_height()[::-1] + (3,))
    return image     



def dos_plotting_singular_real(eps_arr,gf_arr,plot_color,gamma_val,psi):
    
    psi_fig,psi_ax=plt.subplots()
    arr_max=np.amax(np.asarray(gf_arr))               
    eps_boundary=np.where(eps_arr>1.0)[0][0] # where eps>0 starts
    lw=2
    match_impurity_band_with_continuum=True
    
    if match_impurity_band_with_continuum: 
        psi_ax.plot(eps_arr[:eps_boundary],np.asarray(gf_arr[:eps_boundary])/arr_max,
                    color=plot_color,
                    linewidth=lw,
                    label=r"$\gamma= $"+str(np.round(gamma_val,2)))
    else:
        psi_ax.plot(eps_arr[:eps_boundary],np.asarray(gf_arr[:eps_boundary]),
                    color=plot_color,
                    linewidth=lw,
                    label=r"$\gamma= $"+str(np.round(gamma_val,2)))
            
    psi_ax.plot(eps_arr[eps_boundary:],np.asarray(gf_arr[eps_boundary:])/arr_max,
                color=plot_color,
                linewidth=lw)

    yaxis=r"$G^{'}(\epsilon/|\Delta|)$ (arb. units)"
    plotting_spec_singular(psi_fig,psi_ax,yaxis)
    psi_ax.set_title(r"$\psi=$" + str(psi))
    psi_fig.tight_layout()
    psi_fig.savefig("dos_real.pdf",format="pdf")


def dos_plotting_singular_imag(eps_arr,gf_arr,plot_color,gamma_val,psi):
    
    psi_fig,psi_ax=plt.subplots()
    arr_max=np.amax(np.asarray(gf_arr))               
    eps_boundary=np.where(eps_arr>1.0)[0][0] # where eps>0 starts
    lw=2
    match_impurity_band_with_continuum=True
    
    if match_impurity_band_with_continuum: 
        psi_ax.plot(eps_arr[:eps_boundary],np.asarray(gf_arr[:eps_boundary])/arr_max,
                    color=plot_color,
                    linewidth=lw,
                    label=r"$\gamma= $"+str(np.round(gamma_val,2)))
    else:
        psi_ax.plot(eps_arr[:eps_boundary],np.asarray(gf_arr[:eps_boundary]),
                    color=plot_color,
                    linewidth=lw,
                    label=r"$\gamma= $"+str(np.round(gamma_val,2)))
            
    psi_ax.plot(eps_arr[eps_boundary:],np.asarray(gf_arr[eps_boundary:])/arr_max,
                color=plot_color,
                linewidth=lw)
    
    yaxis=r"$G^{''}(\epsilon/|\Delta|)$ (arb. units)"
    plotting_spec_singular(psi_fig,psi_ax,yaxis)
    psi_ax.set_title(r"$\psi=$" + str(psi))
    psi_fig.tight_layout()
    psi_fig.savefig("dos_imag.pdf",format="pdf")


def plotting_spec_singular(psi_fig,psi_ax,psi_y):
    psi_ax.set_ylabel(psi_y)
    psi_ax.set_xlabel(r"$\epsilon/|\Delta|$")
    psi_ax.legend()
    psi_ax.grid()    
    psi_ax.axhspan(-0.2, 1.2, facecolor='#e0dede', alpha=0.2)
    psi_ax.set_ylim(0.0,1.1)
    psi_ax.grid(color="#3f3f44",linestyle='--')    
    psi_fig.tight_layout()
    return psi_fig
    


def plotting_spec_anim(psi_fig,psi_ax):
    psi_ax.set_ylabel(r"$\rho(\epsilon/|\Delta|)/\rho_{0}$ (arb. units)")
    psi_ax.set_xlabel(r"$\epsilon/|\Delta|$")
    psi_ax.legend()
    psi_ax.grid()    
    psi_ax.axhspan(-0.2, 1.2, facecolor='#e0dede', alpha=0.2)
    psi_ax.set_ylim(0.0,1.1)
    psi_ax.grid(color="#3f3f44",linestyle='--')    
    psi_fig.tight_layout()
    return psi_fig
    
def plotting_spec(psi_fig,psi_ax,gamma):
    psi_ax.set_ylabel(r"$\rho(\epsilon/|\Delta|)/\rho_{0}$ (arb. units)")
    psi_ax.set_xlabel(r"$\epsilon/|\Delta|$")
    psi_ax.legend()
    psi_ax.grid()    
    psi_ax.axhspan(-0.2, 1.2, facecolor='#e0dede', alpha=0.2)
    psi_ax.set_ylim(0.0,1.1)
    psi_ax.grid(color="#3f3f44",linestyle='--')    
    psi_ax.set_title(r"$\gamma=$"+str(np.round(gamma,2)))
    psi_fig.tight_layout()
    plot_name="dos_plots/dos_g_"+ str(np.round(gamma,2)) +".pdf"
    print(" --- Saving figure =" + plot_name + " --- ")
    psi_fig.savefig(plot_name,format="pdf")
    print("")


def plot_v_real(eps_values,v_arr,gamma_arr,colors,psi_val):
    font = {'family' : 'sans-serif',
            'weight' : 'normal',
            'size'   : 13}
    matplotlib.rc('font', **font)

    fig,ax=plt.subplots()
    for v,gamma,plot_color in zip(v_arr,gamma_arr,colors):
        ax.plot(eps_values, v,
                label=r"$\gamma$="+str(np.round(gamma,2)),
                linewidth=2,
                color=plot_color)
    ax.legend()
    ax.grid(color="#3f3f44",linestyle='--')
    ax.axhspan(-0.2, 2.1, facecolor='#e0dede', alpha=0.2)
    ax.set_ylim(0.0,2.1)
    ax.set_ylabel(r"$v^{'}(\epsilon/|\Delta|)$")
    ax.set_xlabel(r"$\epsilon/|\Delta|$")
    ax.set_title(r"$\psi= $"+str(psi_val))
    fig.tight_layout()
    fig.savefig("v_real.pdf",format="pdf")
    
def plot_v_imag(eps_values,v_arr,gamma_arr,colors,psi_val):
    font = {'family' : 'sans-serif',
            'weight' : 'normal',
            'size'   : 13}
    matplotlib.rc('font', **font)

    inner_max_arr=[]
    for i in v_arr:
        inner_max=np.amax(np.asarray(i))
        inner_max_arr.append(inner_max)

    maxim=np.asarray(np.amax(inner_max_arr))

    total_v_arr=[]
    for i in v_arr:
        n_v_arr=[]
        for j in i:
            n_v_arr.append(np.abs(j/maxim))
            #            n_v_arr.append(j/maxim)
        total_v_arr.append(n_v_arr)
    
    fig,ax=plt.subplots()
    for v,gamma,plot_color in zip(total_v_arr,gamma_arr,colors):
        ax.plot(eps_values, v,
                label=r"$\gamma$="+str(np.round(gamma,2)),
                linewidth=2,
                color=plot_color)
    ax.legend()
    ax.grid(color="#3f3f44",linestyle='--')
#    ax.axhspan(-0.2, maxim+0.1, facecolor='#e0dede', alpha=0.2)
#    ax.set_ylim(0.0,maxim+0.01)
    ax.set_ylabel(r"$v^{''}(\epsilon/|\Delta|)$")
    ax.set_xlabel(r"$\epsilon/|\Delta|$")
    ax.set_title(r"$\psi= $" +str(psi_val))
    fig.tight_layout()
    fig.savefig("v_imag.pdf",format="pdf")

def plot_h_real():
    print("In plot_h_real")

def plot_h_imag():
    print("In plot_h_real")
    
def fermi_dirac_func(eps,T,ef):
    return 1/(mp.exp((eps-ef)/T)+1)
           
    
def q2b_func(gf,ff,jj_rr,jj_ra,jj_aa):

    prefactor=3./2.

    t1=((ff)**2)*(jj_rr)
    t2=((mp.conj(ff))**2)*(jj_aa)
    t3=(1 + (abs(gf))**2 + (abs(ff))**2)*(jj_ra)

    ret_val=prefactor*(t1+t2+t3)

    return ret_val
    

def jj(l0,v_f,kp):
    prefactor=-mp.j/(v_f*kp)    
    t1=(1-l0**2)*mp.log((l0-1)/(l0+1)) 
    t2=-2*l0
    ret_val=prefactor*(t1+t2)
    return ret_val

    
def l_0(hp,hm,v_f,kp,nm_scatt,typ):

    prefactor=1/(v_f*kp)
    
    if typ == "RA":
        hsum=hp-mp.conj(hm)
    if typ == "RR":
        hsum=hp+hm        
    
    hsum=prefactor*(hsum + (mp.j/nm_scatt))
    return hsum

def qf(res):
    g=270  # https://en.wikipedia.org/wiki/Superconducting_radio_frequency
    q=(g)/(res)
    return q


def q0_func(lambda_L):
    c=3.0E8
    q_0=(4*np.pi)*c/(lambda_L)
    return q_0
    
def reconstruct_gf(gf_r,gf_i):
    gf_new_arr=[]
    for gr,gi in zip(gf_r,gf_i):
        gf_new=gr + mp.j*gi
        gf_new_arr.append(gf_new)
    return gf_new_arr


def plot_dos():

    ang=1E-10
    mega=1E6

    ########################
    kpoint, omega = 0.01, 0.0
    omega2=0.01
    nm_scatt = 0.25
    v_f=1.36 # nb fermi velocity
    gamma=0.1 
    psi_alpha=0.05 
    gamma_arr=[0.9,0.1]
    psi_alpha_arr=[0.1,0.05]
    q0=1.0
    T=0.01
    ef=0.0
    eps_step=0.01
    ########################
    
    font = {'family' : 'sans-serif',
            'weight' : 'normal',
            'size'   : 13}
    matplotlib.rc('font', **font)
    fig,ax=plt.subplots(figsize=(7,5))


    colors=["#00A79D","#BE1E2D"]
    lstyles=["--","-"]
    for gamma,psi_alpha,col,ls in zip(gamma_arr,psi_alpha_arr,colors,lstyles):
        lb=r"$\gamma=$"+str(np.round(1-np.round(gamma,2),2))+r", $\tau_{s}^{-1}=$"+str(np.round(psi_alpha,2))
        eps_arr=construct_eps_arr()
        eps_p_arr=eps_arr+omega/2. # positive branch
        eps_m_arr=eps_arr-omega/2. # negative branch 

        k_arr=construct_k_arr()

        print("Plotting DOS")
        # create gf    
        gf,ff,v=create_gf(eps_arr,gamma,psi_alpha)
        gf_r,gf_i=decompose_gf(gf)
        ff_r,ff_i=decompose_gf(ff)
        gf=reconstruct_gf(gf_r,gf_i)
        ff=reconstruct_gf(ff_r,ff_i)
    
        hf=create_h(eps_arr,gf,ff)
        hf_r,hf_i=decompose_gf(hf)
    
        # create gf positive 
        gf_p,ff_p,v_p=create_gf(eps_p_arr,gamma,psi_alpha)
        gf_p_r,gf_p_i=decompose_gf(gf_p)
        ff_p_r,ff_p_i=decompose_gf(ff_p)
        gf_p=reconstruct_gf(gf_p_r,gf_p_i)
        ff_p=reconstruct_gf(ff_p_r,ff_p_i)
        
        hf_p=create_h(eps_p_arr,gf_p,ff_p)
        hf_p_r,hf_p_i=decompose_gf(hf_p)
        
        # minus
        gf_m,ff_m,v_m=create_gf(eps_m_arr,gamma,psi_alpha)
        gf_m_r,gf_m_i=decompose_gf(gf_m)
        ff_m_r,ff_m_i=decompose_gf(ff_m)
        gf_m=reconstruct_gf(gf_m_r,gf_m_i)
        ff_m=reconstruct_gf(ff_m_r,ff_m_i)
        
        hf_m=create_h(eps_p_arr,gf_m,ff_m)
        hf_m_r,hf_m_i=decompose_gf(hf_m)

        max_val=np.amax(np.asarray(gf_p_r))

        ax.plot(eps_p_arr,np.asarray(gf_p_r)/max_val,
                linewidth=3,
                label=lb,
                color=col,
                linestyle=ls)
        ax.fill_between(eps_p_arr,np.asarray(gf_p_r)/max_val,
                        0,
                        alpha=0.2,
                        color=col)

        ax.legend(loc='upper left')
        ax.set_ylim(-0.05,1.0)
        ax.set_xlim(0.0,2.0)
        ax.set_ylabel(r"$\rho(\epsilon/|\Delta|)/\rho_{0}$ (arb. units)",fontsize=20)
        ax.set_xlabel(r"$\epsilon/|\Delta|$",fontsize=20)
        xlab=[0,0.5,1,1.5,2]

        ax.set_xticks(xlab)
        ax.tick_params(axis='both', which='major', labelsize=20)

    fig.tight_layout()
    plt.show()
    fig.savefig("dos.pdf",format="pdf")

def plot_q2b():
    ang=1E-10
    mega=1E6

    ########################
    kpoint, omega = 0.01, 0.0
    omega2=0.01
    nm_scatt = 0.25
    v_f=1.36 # nb fermi velocity
    gamma=0.1 
    psi_alpha=0.05
    gamma_arr=[0.9,0.1]
    psi_alpha_arr=[0.1,0.05]
    q0=1.0
    T=0.01
    ef=0.0
    eps_step=0.01
    ########################
    
    font = {'family' : 'sans-serif',
            'weight' : 'normal',
            'size'   : 13}
    matplotlib.rc('font', **font)
    fig,ax=plt.subplots(figsize=(7,5))
    # inset for q2b plot n
    inset = inset_axes(ax,
                       width="30%", # width = 30% of parent_bbox
                       height="30%",
                       loc='center left',
                       bbox_to_anchor=(0.1,0,1,1),
                       bbox_transform=ax.transAxes)
    inset.tick_params(labelsize=8)


    colors=["#00A79D","#BE1E2D"]
    lstyles=["--","-"]
    for gamma,psi_alpha,col,ls in zip(gamma_arr,psi_alpha_arr,colors,lstyles):
        lb=r"$\gamma=$"+str(np.round(1-np.round(gamma,2),2))+r", $\tau_{s}^{-1}=$"+str(np.round(psi_alpha,2))
        eps_arr=construct_eps_arr()
        eps_p_arr=eps_arr+omega/2. # positive branch
        eps_m_arr=eps_arr-omega/2. # negative branch 

        k_arr=construct_k_arr()

        print("Plotting Q2B")
        # create gf    
        gf,ff,v=create_gf(eps_arr,gamma,psi_alpha)
        gf_r,gf_i=decompose_gf(gf)
        ff_r,ff_i=decompose_gf(ff)
        gf=reconstruct_gf(gf_r,gf_i)
        ff=reconstruct_gf(ff_r,ff_i)
    
        hf=create_h(eps_arr,gf,ff)
        hf_r,hf_i=decompose_gf(hf)
    
        # create gf positive 
        gf_p,ff_p,v_p=create_gf(eps_p_arr,gamma,psi_alpha)
        gf_p_r,gf_p_i=decompose_gf(gf_p)
        ff_p_r,ff_p_i=decompose_gf(ff_p)
        gf_p=reconstruct_gf(gf_p_r,gf_p_i)
        ff_p=reconstruct_gf(ff_p_r,ff_p_i)
        
        hf_p=create_h(eps_p_arr,gf_p,ff_p)
        hf_p_r,hf_p_i=decompose_gf(hf_p)
        
        # minus 
        gf_m,ff_m,v_m=create_gf(eps_m_arr,gamma,psi_alpha)
        gf_m_r,gf_m_i=decompose_gf(gf_m)
        ff_m_r,ff_m_i=decompose_gf(ff_m)
        gf_m=reconstruct_gf(gf_m_r,gf_m_i)
        ff_m=reconstruct_gf(ff_m_r,ff_m_i)
    
        hf_m=create_h(eps_p_arr,gf_m,ff_m)
        hf_m_r,hf_m_i=decompose_gf(hf_m)
        
        max_val=np.amax(np.asarray(gf_p_r))


        l0_RR_k_arr, l0_RA_k_arr= [], []
        for kpoint in k_arr:
            l0_RR_arr, l0_RA_arr= [], []
            for hp,hm in zip (hf_p,hf_m):
                l0_RR=l_0(hp,hm,v_f,kpoint,nm_scatt,"RR")
                l0_RA=l_0(hp,hm,v_f,kpoint,nm_scatt,"RA")
                l0_RR_arr.append(l0_RR), l0_RA_arr.append(l0_RA)
            l0_RR_k_arr.append(l0_RR_arr), l0_RA_k_arr.append(l0_RA_arr)        


        l0_rr_r_k_arr, l0_rr_i_k_arr = [] , []
        l0_ra_r_k_arr, l0_ra_i_k_arr = [] , []
        for rr,ra in zip(l0_RR_k_arr,l0_RA_k_arr):
            l0_rr_r,l0_rr_i=decompose_gf(rr)
            l0_ra_r,l0_ra_i=decompose_gf(ra)
            l0_rr_r_k_arr.append(l0_rr_r), l0_rr_i_k_arr.append(l0_rr_i)
            l0_ra_r_k_arr.append(l0_ra_r), l0_ra_i_k_arr.append(l0_ra_i)


        jj_RR_k_arr, jj_RA_k_arr, jj_AA_k_arr = [], [], []
        for kpoint in k_arr:
            jj_RR_arr, jj_RA_arr, jj_AA_arr = [], [], []
            for lrr,lra in zip(l0_RR_arr,l0_RA_arr):
                jj_rr=jj(lrr,v_f,kpoint)
                jj_ra=jj(lra,v_f,kpoint)
                jj_aa=mp.conj(jj_rr)
                jj_RR_arr.append(jj_rr), jj_RA_arr.append(jj_ra), jj_AA_arr.append(jj_aa)
            jj_RR_k_arr.append(jj_RR_arr), jj_RA_k_arr.append(jj_RA_arr), jj_AA_k_arr.append(jj_AA_arr)

        rr_r_k_arr, rr_i_k_arr = [] , []
        ra_r_k_arr, ra_i_k_arr = [] , []
        aa_r_k_arr, aa_i_k_arr = [] , []
        for rr,ra,aa in zip(jj_RR_k_arr,jj_RA_k_arr,jj_AA_k_arr):
            rr_r,rr_i=decompose_gf(rr)
            ra_r,ra_i=decompose_gf(ra)
            aa_r,aa_i=decompose_gf(aa)
            rr_r_k_arr.append(rr_r), rr_i_k_arr.append(rr_i)
            ra_r_k_arr.append(ra_r), ra_i_k_arr.append(ra_i)
            aa_r_k_arr.append(aa_r), aa_i_k_arr.append(aa_i)        

        # #### eq 5.3 ####
        print("Eq 5.3")
        q2b_k_arr = []
        for rrk,rak,aak in zip(jj_RR_k_arr,jj_RA_k_arr,jj_AA_k_arr):
            q2b_arr = []
            for g,f,rr,ra,aa in zip(gf_p,ff_p,rrk,rak,aak):
                q2b=q2b_func(g,f,rr,ra,aa)
                q2b_arr.append(q2b)
            q2b_k_arr.append(q2b_arr)

        q2b_r_k_arr, q2b_i_k_arr = [] , []
        for q in q2b_k_arr:
            q2b_r,q2b_i=decompose_gf(q)
            q2b_r_k_arr.append(q2b_r)
            q2b_i_k_arr.append(q2b_i)


        for kp,q in zip(k_arr,q2b_r_k_arr):
            max_val=np.amax(np.asarray(q))
            ax.plot(eps_arr,q/max_val,
                    linewidth=3,
                    label=lb,
                    color=col,
                    linestyle=ls)            
            ax.fill_between(eps_arr,q/max_val,
                            0,
                            alpha=0.2,
                            color=col)

        ax.legend(loc='upper left')
        ymax=np.amax(np.asarray(q2b_r_k_arr))
        ax.set_xlim(0.0,2.0)
        ax.set_ylim(-0.05,1.0)
        xlab=[0,0.5,1,1.5,2]
        ax.set_xticks(xlab)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_ylabel(r"$\bar{Q}_{2}(\varepsilon,k)$",fontsize=20)
        ax.set_xlabel(r"$\epsilon/|\Delta|$",fontsize=20)
        inset_min=int(0.001*len(np.asarray(eps_arr)))
        inset_max=int(0.25*len(np.asarray(eps_arr)))
        print(inset_min,inset_max,len(np.asarray(eps_arr)))
        interval=1
        inset.plot(np.asarray(eps_arr[inset_min:inset_max:interval]),
                   q[inset_min:inset_max:interval]/max_val,
                   linestyle=ls,
                   color=col)
        inset.fill_between(np.asarray(eps_arr[inset_min:inset_max:interval]),
                           q[inset_min:inset_max:interval]/max_val,
                           0,
                           alpha=0.2,
                           color=col)
    fig.tight_layout()  
    plt.show()
    fig.savefig("q2b.pdf",format="pdf")


def plot_supp():
    print("Supplementary plot")
    ang=1E-10
    mega=1E6

    ########################
    kpoint, omega = 0.01, 0.0
    omega2=0.01
    nm_scatt = 0.25
    v_f=1.36 # nb fermi velocity
    gamma=0.1 
    psi_alpha=0.05 
    gamma_arr=[0.9,0.8,0.7]
    psi_alpha_arr=[0.1,0.25,0.5]
    q0=1.0
    T=0.01
    ef=0.0
    eps_step=0.01
    ########################
    
    font = {'family' : 'sans-serif',
            'weight' : 'normal',
            'size'   : 13}
    matplotlib.rc('font', **font)
    fig,ax=plt.subplots(figsize=(7,5))
    colors=["#00A79D","#0019a7","#7a00a7"]
    
    lstyles=["--","--","--"]
    for gamma,psi_alpha,col,ls in zip(gamma_arr,psi_alpha_arr,colors,lstyles):
        lb=r"$\gamma=$"+str(np.round(1-np.round(gamma,2),2))+r", $\tau_{s}^{-1}=$"+str(np.round(psi_alpha,2))
        eps_arr=construct_eps_arr()
        eps_p_arr=eps_arr+omega/2. # positive branch
        eps_m_arr=eps_arr-omega/2. # negative branch 

        k_arr=construct_k_arr()

        gf,ff,v=create_gf(eps_arr,gamma,psi_alpha)
        gf_r,gf_i=decompose_gf(gf)
        ff_r,ff_i=decompose_gf(ff)
        gf=reconstruct_gf(gf_r,gf_i)
        ff=reconstruct_gf(ff_r,ff_i)
    
        hf=create_h(eps_arr,gf,ff)
        hf_r,hf_i=decompose_gf(hf)
    
        # create gf positive 
        gf_p,ff_p,v_p=create_gf(eps_p_arr,gamma,psi_alpha)
        gf_p_r,gf_p_i=decompose_gf(gf_p)
        ff_p_r,ff_p_i=decompose_gf(ff_p)
        gf_p=reconstruct_gf(gf_p_r,gf_p_i)
        ff_p=reconstruct_gf(ff_p_r,ff_p_i)
        
        hf_p=create_h(eps_p_arr,gf_p,ff_p)
        hf_p_r,hf_p_i=decompose_gf(hf_p)
        
        # minus 
        gf_m,ff_m,v_m=create_gf(eps_m_arr,gamma,psi_alpha)
        gf_m_r,gf_m_i=decompose_gf(gf_m)
        ff_m_r,ff_m_i=decompose_gf(ff_m)
        gf_m=reconstruct_gf(gf_m_r,gf_m_i)
        ff_m=reconstruct_gf(ff_m_r,ff_m_i)
    
        hf_m=create_h(eps_p_arr,gf_m,ff_m)
        hf_m_r,hf_m_i=decompose_gf(hf_m)
        
        max_val=np.amax(np.asarray(gf_p_r))

        l0_RR_k_arr, l0_RA_k_arr= [], []
        for kpoint in k_arr:
            l0_RR_arr, l0_RA_arr= [], []
            for hp,hm in zip (hf_p,hf_m):
                l0_RR=l_0(hp,hm,v_f,kpoint,nm_scatt,"RR")
                l0_RA=l_0(hp,hm,v_f,kpoint,nm_scatt,"RA")
                l0_RR_arr.append(l0_RR), l0_RA_arr.append(l0_RA)
            l0_RR_k_arr.append(l0_RR_arr), l0_RA_k_arr.append(l0_RA_arr)        


        l0_rr_r_k_arr, l0_rr_i_k_arr = [] , []
        l0_ra_r_k_arr, l0_ra_i_k_arr = [] , []
        for rr,ra in zip(l0_RR_k_arr,l0_RA_k_arr):
            l0_rr_r,l0_rr_i=decompose_gf(rr)
            l0_ra_r,l0_ra_i=decompose_gf(ra)
            l0_rr_r_k_arr.append(l0_rr_r), l0_rr_i_k_arr.append(l0_rr_i)
            l0_ra_r_k_arr.append(l0_ra_r), l0_ra_i_k_arr.append(l0_ra_i)

        jj_RR_k_arr, jj_RA_k_arr, jj_AA_k_arr = [], [], []
        for kpoint in k_arr:
            jj_RR_arr, jj_RA_arr, jj_AA_arr = [], [], []
            for lrr,lra in zip(l0_RR_arr,l0_RA_arr):
                jj_rr=jj(lrr,v_f,kpoint)
                jj_ra=jj(lra,v_f,kpoint)
                jj_aa=mp.conj(jj_rr)
                jj_RR_arr.append(jj_rr), jj_RA_arr.append(jj_ra), jj_AA_arr.append(jj_aa)
            jj_RR_k_arr.append(jj_RR_arr), jj_RA_k_arr.append(jj_RA_arr), jj_AA_k_arr.append(jj_AA_arr)

        rr_r_k_arr, rr_i_k_arr = [] , []
        ra_r_k_arr, ra_i_k_arr = [] , []
        aa_r_k_arr, aa_i_k_arr = [] , []
        for rr,ra,aa in zip(jj_RR_k_arr,jj_RA_k_arr,jj_AA_k_arr):
            rr_r,rr_i=decompose_gf(rr)
            ra_r,ra_i=decompose_gf(ra)
            aa_r,aa_i=decompose_gf(aa)
            rr_r_k_arr.append(rr_r), rr_i_k_arr.append(rr_i)
            ra_r_k_arr.append(ra_r), ra_i_k_arr.append(ra_i)
            aa_r_k_arr.append(aa_r), aa_i_k_arr.append(aa_i)        

        q2b_k_arr = []
        for rrk,rak,aak in zip(jj_RR_k_arr,jj_RA_k_arr,jj_AA_k_arr):
            q2b_arr = []
            for g,f,rr,ra,aa in zip(gf_p,ff_p,rrk,rak,aak):
                q2b=q2b_func(g,f,rr,ra,aa)
                q2b_arr.append(q2b)
            q2b_k_arr.append(q2b_arr)

        q2b_r_k_arr, q2b_i_k_arr = [] , []
        for q in q2b_k_arr:
            q2b_r,q2b_i=decompose_gf(q)
            q2b_r_k_arr.append(q2b_r)
            q2b_i_k_arr.append(q2b_i)


        for kp,q in zip(k_arr,q2b_r_k_arr):
            max_val=np.amax(np.asarray(q))
            ax.plot(eps_arr,q/max_val,
                    linewidth=3,
                    label=lb,
                    color=col,
                    linestyle=ls)            
            ax.fill_between(eps_arr,q/max_val,
                            0,
                            alpha=0.1,
                            color=col)

        ax.legend(loc='upper left')
        ymax=np.amax(np.asarray(q2b_r_k_arr))
        ax.set_xlim(0.0,2.0)
        ax.set_ylim(-0.05,1.0)
        xlab=[0,0.5,1,1.5,2]
        ax.set_xticks(xlab)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_ylabel(r"$\bar{Q}_{2}(\varepsilon,k)$",fontsize=20)
        ax.set_xlabel(r"$\epsilon/|\Delta|$",fontsize=20)
    fig.tight_layout()    
    plt.show()
    fig.savefig("q2b.pdf",format="pdf")

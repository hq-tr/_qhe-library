# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:20:04 2021

@author: User
"""
import numpy as np
import misc
from sympy.physics.quantum.cg import CG
from scipy.sparse import *
from itertools import combinations

def ladder_coefficient(S,M,mode="+"):
    assert mode in ["+","-"]
    if mode=="+":
        term = (S-M)*(S+M+1)
    elif mode=="-":
        term = (S+M)*(S-M+1)
    print(f"sqrt({term})")
    return np.sqrt(term)
    
def Lminus_monomial(basis, Np, basis_format="dec"):
    S = Np/2-1
    if basis_format=="dec":
        basis_index = misc.dec_to_index(basis)
    elif basis_format=="index":
        basis_index = basis
    Ne = len(basis_index)
    output = {}
    for i in range(Ne):
        new = [basis_index[x]-1 if x==i else basis_index[x] for x in range(Ne)]
        m = basis_index[i]
        coef = np.sqrt((S+m)*(S-m+1))
        output[misc.index_to_decimal(new)]=coef
    
    return output

def differentiator(basis_list,coef_list, Np, debug=False,partial=0):
   # S = (Np-1)/2
    assert len(basis_list)==len(coef_list), "Basis and coefficient sizes must be equal"
    dim = len(basis_list)
    output = {}
       
    for i in range(dim):
        basis_index = misc.dec_to_index(basis_list[i])
        if debug: print(f"________{basis_index}")
        Ne = len(basis_index)
        print(Ne)
        if partial == 0:
            Ne_proc = Ne
        else:
            assert 0 < partial < Ne
            Ne_proc = partial
        for j in range(Ne_proc):
            new = np.array([basis_index[x]-1 if x==j else basis_index[x] for x in range(Ne)])
            if debug: print(new,end="\t")
            if not(-1 in new or 0 in new[:-1]-new[1:]):
                new_dec = misc.index_to_decimal(new)
                #m = basis_index[j]-S
                #if debug: print(m,end="\t")
                coef = basis_index[j]*coef_list[i]
                if debug: print(coef)
                if new_dec in output:
                    output[new_dec]+=coef
                else:
                    output[new_dec]=coef
            else:
                if debug: print("skipped")
    if debug: print(output,flush=1)
    norm = np.sqrt(sum(output[x]*np.conj(output[x]) for x in output))
    if debug: print(f"Norm = {norm}")
    out_vec = np.array([x for x in output])
    if norm>1e-13:
        out_coef = np.array([output[x] for x in output])
    else:
        out_coef = np.zeros(len(output))
    return out_vec, out_coef

def Lminus(basis_list,coef_list, Np, debug=False,partial=0,normalize=True):
    S = (Np-1)/2
    assert len(basis_list)==len(coef_list), "Basis and coefficient sizes must be equal"
    dim = len(basis_list)
    output = {}
       
    for i in range(dim):
        basis_index = misc.dec_to_index(basis_list[i])
        if debug: print(f"________{basis_index}")
        Ne = len(basis_index)
        if debug: print(Ne)
        if partial == 0:
            Ne_proc = Ne
        else:
            assert 0 < partial < Ne
            Ne_proc = partial
        for j in range(Ne_proc):
            new = np.array([basis_index[x]-1 if x==j else basis_index[x] for x in range(Ne)])
            if debug: print(new,end="\t")
            if not(-1 in new or 0 in new[:-1]-new[1:]):
                new_dec = misc.index_to_decimal(new)
                m = basis_index[j]-S
                if debug: print(m,end="\t")
                coef = np.sqrt((S+m)*(S-m+1))*coef_list[i]
                if debug: print(coef)
                if new_dec in output:
                    output[new_dec]+=coef
                else:
                    output[new_dec]=coef
            else:
                if debug: print("skipped")
    if debug: print(output,flush=1)
    norm = np.sqrt(sum(output[x]*np.conj(output[x]) for x in output))
    if debug: print(f"Norm = {norm}")
    out_vec = np.array([x for x in output])
    if norm>1e-13:
        if normalize: 
            out_coef = np.array([output[x]/norm for x in output])
        else:
            out_coef = np.array([output[x] for x in output])
    else:
        out_coef = np.zeros(len(output))
    return out_vec, out_coef

def Lminus_boson_defuct(basis_list,coef_list, Np, debug=False,partial=0,normalize=True):
    print("Warning: This routine doesn't work lol")
    S = (Np-1)/2
    assert len(basis_list)==len(coef_list), "Basis and coefficient sizes must be equal"
    dim = len(basis_list)
    output = {}
       
    for i in range(dim):
        basis_index = basis_list[i]
        if debug: print(f"________{basis_index}")
        Ne = len(basis_index)
        #print(Ne)
        if partial == 0:
            Ne_proc = Ne
        else:
            assert 0 < partial < Ne
            Ne_proc = partial
        for j in range(Ne_proc):
            new = np.array([basis_index[x]-1 if x==j else basis_index[x] for x in range(Ne)])
            if debug: print(new,end="\t")
            if not -1 in new:
                #new_dec = misc.index_to_decimal(new)
                m = basis_index[j]-S
                if debug: print(m,end="\t")
                coef = np.sqrt((S+m)*(S-m+1))*coef_list[i]
                if debug: print(coef)
                if new in output:
                    output[new]+=coef
                else:
                    output[new]=coef
            else:
                if debug: print("skipped")
    if debug: print(output,flush=1)
    norm = np.sqrt(sum(output[x]*np.conj(output[x]) for x in output))
    print(f"Norm = {norm}")
    out_vec = np.array([x for x in output])
    if norm>1e-13:
        if normalize:
            out_coef = np.array([output[x]/norm for x in output])
        else:
            out_coef = np.array([output[x] for x in output])
    else:
        out_coef = np.zeros(len(output))
    return out_vec, out_coef

def Lplus(basis_list,coef_list, Np, debug=False, partial=0,normalize=True):
    S = (Np-1)/2
    assert len(basis_list)==len(coef_list), "Basis and coefficient sizes must be equal"
    dim = len(basis_list)
    output = {}
    for i in range(dim):
        basis_index = misc.dec_to_index(basis_list[i])
        if debug: print(f"________{basis_index}")
        Ne = len(basis_index)
        for j in range(Ne):
            new = np.array([basis_index[x]+1 if x==j else basis_index[x] for x in range(Ne)])
            if debug: print(new,end="\t")
            if not(Np in new or 0 in new[:-1]-new[1:]):
                new_dec = misc.index_to_decimal(new)
                m = basis_index[j]-S
                if debug: print(m,end="\t")
                coef = np.sqrt((S-m)*(S+m+1))*coef_list[i]
                if debug: print(coef)
                if new_dec in output:
                    output[new_dec]+=coef
                else:
                    output[new_dec]=coef
            else:
                if debug: print("skipped")
    if debug: print(output,flush=1)
    norm = np.sqrt(sum(output[x]*np.conj(output[x]) for x in output))
    if debug: print(f"Norm = {norm}")
    out_vec = np.array([x for x in output])
    if norm>1e-20:
        if normalize:
            out_coef = np.array([output[x]/norm for x in output])
        else:
            out_coef = np.array([output[x] for x in output])
    else:
        out_coef = np.zeros(len(output))
    return out_vec, out_coef

def Lplus_boson(basis_list,coef_list, debug=False, partial=0,normalize=True):
    Np = len(basis_list[0])
    S = (Np-1)/2
    assert len(basis_list)==len(coef_list), "Basis and coefficient sizes must be equal"
    dim = len(basis_list)
    output = {}
    for i in range(dim):
        basis_index = misc.binary_to_index_boson(basis_list[i])
        basis_multi = [basis_index.count(x) for x in basis_index]
        if debug: print(f"________{basis_index}\t{basis_multi}")
        Ne = len(basis_index)
        if partial == 0:
            Ne_proc = Ne
        else:
            assert 0 < partial < Ne
            Ne_proc = partial
        for j in range(Ne_proc):
            new = [basis_index[x]+1 if x==j else basis_index[x] for x in range(Ne)]
            new.sort()
            if debug: print(new,end="\t")
            if not Np in new:
                new_bin = misc.index_to_binary_boson(new,Np)
                m = basis_index[j]-S
                if debug: print(m,end="\t")
                multi_new = new.count(basis_index[j]+1)
                if debug: print(multi_new, end="\t")
                coef = np.sqrt((S-m)*(S+m+1))*coef_list[i]*np.sqrt(multi_new)/np.sqrt(basis_multi[j])
                if debug: print(f"{basis_multi[j]}\t{coef}")
                if new_bin in output:
                    output[new_bin]+=coef
                else:
                    output[new_bin]=coef
            else:
                if debug: print("skipped")
    if debug: print(output,flush=1)
    norm = np.sqrt(sum(output[x]*np.conj(output[x]) for x in output))
    if debug: print(f"Norm = {norm}")
    out_vec = np.array([x for x in output])
    if norm>1e-13:
        if normalize:
            out_coef = np.array([output[x]/norm for x in output])
        else:
            out_coef = np.array([output[x] for x in output])
    else:
        out_coef = np.zeros(len(output))
    return out_vec, out_coef

def Lminus_boson(basis_list,coef_list, debug=False, partial=0,normalize=True):
    Np = len(basis_list[0])
    S = (Np-1)/2
    assert len(basis_list)==len(coef_list), "Basis and coefficient sizes must be equal"
    dim = len(basis_list)
    output = {}
    for i in range(dim):
        basis_index = misc.binary_to_index_boson(basis_list[i])
        basis_multi = [basis_index.count(x) for x in basis_index]
        if debug: print(f"________{basis_index}\t{basis_multi}")
        Ne = len(basis_index)
        if partial == 0:
            Ne_proc = Ne
        else:
            assert 0 < partial < Ne
            Ne_proc = partial
        for j in range(Ne_proc):
            new = [basis_index[x]-1 if x==j else basis_index[x] for x in range(Ne)]
            new.sort()
            if debug: print(new,end="\t")
            if not -1 in new:
                new_bin = misc.index_to_binary_boson(new,Np)
                m = basis_index[j]-S
                if debug: print(m,end="\t")
                multi_new = new.count(basis_index[j]-1)
                if debug: print(multi_new, end="\t")
                coef = np.sqrt((S+m)*(S-m+1))*coef_list[i]*np.sqrt(multi_new)/np.sqrt(basis_multi[j])
                if debug: print(f"{basis_multi[j]}\t{coef}")
                if new_bin in output:
                    output[new_bin]+=coef
                else:
                    output[new_bin]=coef
            else:
                if debug: print("skipped")
    if debug: print(output,flush=1)
    norm = np.sqrt(sum(output[x]*np.conj(output[x]) for x in output))
    if debug: print(f"Norm = {norm}")
    out_vec = np.array([x for x in output])
    if norm>1e-13:
        if normalize:
            out_coef = np.array([output[x]/norm for x in output])
        else:
            out_coef = np.array([output[x] for x in output])
    else:
        out_coef = np.zeros(len(output))
    return out_vec, out_coef

def take_difference(vec1_index,vec2_index,take_abs = True):
    dim = len(vec1_index)
    assert len(vec2_index) == dim, "vectors should be of the same sizes"
    if take_abs:
        return [abs(vec1_index[i] - vec2_index[i]) for i in range(dim)]
    else:
        return [vec1_index[i] - vec2_index[i] for i in range(dim)]

def LplusLminus(basis_list_index,Np,debug=False):
    dim = len(basis_list_index)
    LL = dok_matrix((dim,dim), dtype=float)
    basis_list_dec=[misc.index_to_decimal(vec) for vec in basis_list_index]
    for (i,j) in combinations(range(dim),2):
        if sum(take_difference(basis_list_index[i],basis_list_index[j]))==2:
            dummy = Lminus([basis_list_dec[i]],[1.0],Np,normalize=False)
            dummy2 = Lminus([basis_list_dec[j]],[1.0],Np,normalize=False)
            basis, vec1, vec2 = misc.collate_vectors(dummy[0],dummy[1],dummy2[0],dummy2[1])
            term = np.dot(vec1, vec2)
            LL[i,j] = term
            LL[j,i] = term
            if debug: print(f"{i}\t{j}\t{term:.8f}")
    for i in range(dim):  #diagonal terms
        dummy = Lminus([basis_list_dec[i]],[1.0],Np,normalize=False)
        LL[i,i] = np.dot(dummy[1],dummy[1])
    return LL.tocsc() 

def LminusLplus_boson(basis_list_index,Np,debug=False):
    dim = len(basis_list_index)
    LL = dok_matrix((dim,dim), dtype=float)
    basis_list_binary=[misc.index_to_binary_boson(vec,Np) for vec in basis_list_index]
    for (i,j) in combinations(range(dim),2):
        if sum(take_difference(basis_list_index[i],basis_list_index[j]))==2:
            dummy = Lplus_boson([basis_list_binary[i]],[1.0],debug=debug,normalize=False)
            dummy2 = Lplus_boson([basis_list_binary[j]],[1.0],debug=debug,normalize=False)
            basis, vec1, vec2 = misc.collate_vectors(dummy[0],dummy[1],dummy2[0],dummy2[1])
            term = np.dot(vec1, vec2)
            LL[i,j] = term
            LL[j,i] = term
            if debug: print(f"{i}\t{j}\t{term:.8f}")
    for i in range(dim):  #diagonal terms
        dummy = Lplus_boson([basis_list_binary[i]],[1.0],debug=debug,normalize=False)
        LL[i,i] = np.dot(dummy[1],dummy[1])
    return LL.tocsc() 

def CG_transform(m1,m2,l1,l2,vec):
    M = m1+m2
    if M > l1+l2: return 0
    n_rep = int(l1+l2-M+1)
    cg_list = np.zeros(n_rep)
    for i in range(n_rep):
        L = M+i
        #print(f"{l1} {m1} {l2} {m2} {L} {M} {CG(l1,m1,l2,m2,L,M).doit()}")
        cg_list[i]=float(CG(l1,m1,l2,m2,L,M).doit())
        print(i,end="\t")
        print(CG(l1,m1,l2,m2,L,M).doit(), end="\t")
        print(vec[i])
    try:
        outvec = np.dot(cg_list,vec)
        norm = np.sqrt(np.dot(outvec,outvec))
        print(norm)
        outvec /= norm
    except Exception as errmsg:
        print(errmsg)
        outvec = np.zeros(vec.shape[1])
    finally:
        return outvec

def Lx(basis,vec,Np, normalize=True): # Lx = (Lplus + Lminus)  / 2
    vecp = Lplus(basis,vec,Np,normalize=False)
    vecm = Lminus(basis,vec,Np,normalize=False)
    
    basis_out, vec1_out, vec2_out = misc.collate_vectors(vecp[0],vecp[1],vecm[0],vecm[1])
    coef_out = np.array(vec1_out) + np.array(vec2_out)
    if normalize:
        coef_out /= np.sqrt(np.dot(coef_out,coef_out))
    else:
        coef_out /= 2
    return basis_out, coef_out

def Ly(basis,vec,Np, normalize=True): # Ly = (Lplus - Lminus)  / 2i
    vecp = Lplus(basis,vec,Np,normalize=False)
    vecm = Lminus(basis,vec,Np,normalize=False)
    
    basis_out, vec1_out, vec2_out = misc.collate_vectors(vecp[0],vecp[1],vecm[0],vecm[1])
    coef_out = np.array(vec1_out) - np.array(vec2_out)
    if normalize:
        coef_out /= np.sqrt(np.dot(coef_out,coef_out))*(1j)
    else:
        coef_out /= 2
    return basis_out, coef_out

def rot_x(basis,vec, Np, theta, maxpow=5):
    outbas = basis
    outvec = vec
    for i in range(maxpow):
        a = Lx(outbas,outvec,Np, normalize=False)
        outbas, vec1, vec2 = misc.collate_vectors(outbas, outvec, a[0],a[1])
        outvec = np.array(vec1)+(1j*theta/2)**(i+1)*np.array(vec2)/np.prod(np.arange(i+1)+1)
    outvec /= np.sqrt(np.dot(outvec,outvec))
    return outbas, outvec

def rot_y(basis,vec, Np, theta, maxpow=5):
    outbas = basis
    outvec = vec
    for i in range(maxpow):
        a = Ly(outbas,outvec,Np, normalize=False)
        outbas, vec1, vec2 = misc.collate_vectors(outbas, outvec, a[0],a[1])
        outvec = np.array(vec1)+(1j*theta/2)**(i+1)*np.array(vec2)/np.prod(np.arange(i+1)+1)
    outvec /= np.sqrt(np.dot(outvec,outvec))
    return outbas, outvec

def Lz(basis, vec, Np, normalize=True, debug = False):
    assert len(basis)==len(vec)
    S = (Np-1)/2
    if debug: print(f"S = {S}")
    Ne = len(misc.dec_to_index(basis[0]))
    if debug: print(f"N_e = {Ne}")
    m = np.array([sum(misc.dec_to_index(x))-S*Ne for x in basis])
    if debug: print(m)
    vecout = vec*m   
    if normalize:
        vecout /= np.sqrt(np.dot(vecout,vecout))
    return basis, vecout

def rot_z(basis,vec, Np, theta, maxpow=5):
    outbas = basis
    outvec = vec
    for i in range(maxpow):
        a = Lz(outbas,outvec,Np, normalize=False)
        outbas, vec1, vec2 = misc.collate_vectors(outbas, outvec, a[0],a[1])
        outvec = np.array(vec1)+(1j*theta/2)**(i+1)*np.array(vec2)/np.prod(np.arange(i+1)+1)
    print(np.sqrt(np.dot(outvec,outvec)))
    outvec /= np.sqrt(np.dot(outvec,outvec))
    return outbas, outvec

if __name__=="__main__":
    ext = {1: "Lp", 2: "Lm"}
    choice = int(input("Apply L_plus (1) or L_minus (2)? "))
    times = int(input("Apply how many times? "))
    filename = str(input("Input file name: "))
    with open(filename) as f:
        dim = int(f.readline())
        a = f.readlines()
        basis_list = list(map(int,a[::2]))
        coef_list  = list(map(float, a[1::2]))
    Np = int(input("Input n_orb: "))
    saveyn = int(input("Save file(s)? (1 for yes)" ))
    for t in range(times):
        if choice == 2:
            outvec, outcoef = Lminus(basis_list,coef_list,Np)
        elif choice == 1:
            outvec, outcoef = Lplus(basis_list,coef_list,Np)
        if saveyn:
            with open(filename+f"_{t+1}{ext[choice]}","w+") as f:
                f.write(f"{len(outvec)}\n")
                towrite = "".join(f"{outvec[x]}\n{outcoef[x]}\n" for x in range(len(outvec)))
                f.write(towrite)
        del basis_list, coef_list
        basis_list = outvec
        coef_list = outcoef
        print(f"Done {t} times",end="\r")
    
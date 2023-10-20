# O primeiro passo é o seguinte:

import math
import scipy
import numpy as np
import pandas as pd
from scipy import interpolate 
from scipy.optimize import nnls,lsq_linear
import matplotlib.pyplot as plt 

#Passo 0 : Importar os dados do arquivo txt
#Criar as matrizes tK e TK que serão usadas futuramente Definição dos parâmetros das amostras como diametro e altura para calculo da sesseção transversal da amostra

l = float(input("Digite o valor de 'l'  da amostra em metros: \n"))
d = float(input("Digite o valor de 'd' da amostra em metros: \n")) 
Rref = float(input("Digite o valor de Rref: \n")) 
nome = str(input("Digite o nome de sua amostra: \n"))
#l=0.0974
#d= 0.0252
#Rref = 2300
A = np.pi*(d/2)**2
tK = np.logspace(-4,3,700)
TK = np.transpose(tK)
n = len(TK)
H = 10 # Verificar o peso no artigo
h = 10 # Verificar o peso no artigo 

# Passo 1 : Obtenção de Resistividade real e imaginária após leitura do arquivo


caminho = input("Digite o caminho do arquivo: \n")
arquivo = pd.read_csv(caminho, header = None, sep = '\t')

# Agora que conseguimos ler o arquivo e separar corretamente as linhas podemos começar os calculos

F = np.array(arquivo[0])
PHI = np.array(arquivo[1])
MAG = np.array(arquivo[2])
w = 2*np.pi*F # Equação 4
p = len(w)
R = Rref*(10**(MAG/20)) #Equação 5
RHO = R*A/l # Equação 6
Rho = RHO * np.cos(PHI) - (1*np.array(0+1j)*RHO*np.sin(PHI)) # Equação 7



G1 = np.zeros((p,n))
G2 = np.zeros((p,n))

def Debye(F,PHI,MAG,Rref,l,A,w,p,R,RHO,Rho):
  for i in range(p):
    for j in range(n):
      g1 = (w[i]*TK[j])**1
      g2 = w[i]*TK[j]
      G1[i,j] = g1/(1+g1)
      G2[i,j] = g2/(1+g1)

  mG2 = nnls(G2,Rho_N_IM)
  mG1 = nnls(G1,Rho_N_RE)
  soma_mG1 = np.sum(mG1[0])
  soma_mG2 = np.sum(mG2[0]) # Diferença no vetor

    #Passo 4
    #Obtenção de RHO_0

  RHO_0 = DC * (1+soma_mG2 - soma_mG1) # Diferença no Rho_0 por conta do vetor mG2

   #Passo 5
    #Correção e normalização da resistividade real e imaginária

  RHO_N_RE = abs((np.real(Rho)-RHO_0))/RHO_0 
  RHO_N_IM = np.imag(Rho)/RHO_0

    #Passo 6
    #Obtenção de MK

  wt = H* np.sum(RHO_N_RE)/np.sum(RHO_N_IM) # Diferença no valor por conta de mG2
  GG1 = np.zeros((p,n))
  GG2 = np.zeros((p,n))
  for i in range(p):
    GG1[i] = G1[i]
    GG2[i] = wt*G2[i]

  d1 = RHO_N_RE
  d2 = wt*RHO_N_IM
  G = np.concatenate((GG1,GG2))
  d = np.concatenate((d1,d2))
  MK = []
  MK = nnls(G,d)

    #Passo 7
    #Obtenção de M e TAU

  cargabilidade = np.sum(MK[0])
  MT = cargabilidade
  MN = MT/RHO_0 # Cargabilidade normalizada
  Ln = np.log(TK)
  Tau = np.exp(np.sum(MK[0]*Ln )/MT)
  TAU = Tau # Tempo de relaxação
  RHO_DC=RHO_0 # Resistividade DC
  SIG_0 =1/RHO_0 # Condutividade DC

    #Passo 8
    #Calculo SIP


  Rho_C_RE = np.zeros((p,1))
  Rho_C_IM = np.zeros((p,1))
  rho_c_re = np.zeros((n,1))
  rho_c_im = np.zeros((n,1))
  Rhoc = np.zeros((p,1))

  for j in range(p):
    w = 2*np.pi*F[j]
    for i in range(n):
      rho_c_re[i]=MK[0][i]*((w*TK[i])**1/(1+(w*TK[i])**1))
      rho_c_im[i]=MK[0][i]*w*TK[i]/(1+(w*TK[i])**1)
    soma1 = np.sum(rho_c_re)
    soma2 = np.sum(rho_c_im)
    Rho_C_RE[j] = soma1
    Rho_C_IM[j] = soma2
  ONE = np.ones((p,1))
  Rhoc = RHO_0*((ONE-Rho_C_RE)-np.array(0+1j)*Rho_C_IM)
  PHIc = np.angle(Rhoc)
  return F,PHI,PHIc,TK,MK,RHO,Rhoc

def ColeCole (c,F,PHI,MAG,Rref,l,A,w,p,R,RHO,Rho):
    for i in range(p):
      for j in range(n):
        g1 = (w[i]*TK[j])**c
        g2 = w[i]*TK[j]
        G1[i,j] = g1/(1+g1)
        G2[i,j] = g2/(1+g1)

    mG2 = nnls(G2,Rho_N_IM)
    mG1 = nnls(G1,Rho_N_RE)
    soma_mG1 = np.sum(mG1[0])
    soma_mG2 = np.sum(mG2[0]) # Diferença no vetor

    # Passo 4  Obtendo RHO_0

    RHO_0 = DC * (1+soma_mG2 - soma_mG1) # Diferença no Rho_0 por conta do vetor mG2


    # Passo 5 - Correção normalização da resistividade real e imaginaria


    RHO_N_RE = abs((np.real(Rho)-RHO_0))/RHO_0 
    RHO_N_IM = np.imag(Rho)/RHO_0


    # Passo 6  Obtenção de MK


    wt = H* np.sum(RHO_N_RE)/np.sum(RHO_N_IM) # Diferença no valor por conta de mG2
    GG1 = np.zeros((p,n))
    GG2 = np.zeros((p,n))
    for i in range(p):
      GG1[i] = G1[i]
      GG2[i] = wt*G2[i]

    d1 = RHO_N_RE
    d2 = wt*RHO_N_IM
    G = np.concatenate((GG1,GG2))
    d = np.concatenate((d1,d2))
    MK = []
    MK = nnls(G,d)

    # Passo 7 Obtenção de M e TAU


    cargabilidade = np.sum(MK[0])
    MT = cargabilidade
    MN = MT/RHO_0 # Cargabilidade normalizada
    Ln = np.log(TK)
    Tau = np.exp(np.sum(MK[0]*Ln )/MT)
    TAU = Tau # Tempo de relaxação
    RHO_DC=RHO_0 # Resistividade DC
    SIG_0 =1/RHO_0 # Condutividade DC


    # Passo 8 Calculo SIP


    Rho_C_RE = np.zeros((p,1))
    Rho_C_IM = np.zeros((p,1))
    rho_c_re = np.zeros((n,1))
    rho_c_im = np.zeros((n,1))
    Rhoc = np.zeros((p,1))

    for j in range(p):
      w = 2*np.pi*F[j]
      for i in range(n):
        rho_c_re[i]=MK[0][i]*((w*TK[i])**c/(1+(w*TK[i])**c))
        rho_c_im[i]=MK[0][i]*w*TK[i]/(1+(w*TK[i])**c)
      soma1 = np.sum(rho_c_re)
      soma2 = np.sum(rho_c_im)
      Rho_C_RE[j] = soma1
      Rho_C_IM[j] = soma2
    ONE = np.ones((p,1))
    Rhoc = RHO_0*((ONE-Rho_C_RE)-np.array(0+1j)*Rho_C_IM)
    PHIc = np.angle(Rhoc)

    return F,PHI,PHIc,TK,MK,RHO,Rhoc


metodo = int(input("Qual(s) método(s) gostaria de usar:\n(1) Debye \n(2) Cole-Cole\n(3) Ambos\n"))
if metodo == 1:
  F_d,PHI_d,PHIc_d,TK_d,MK_d,RHO_d,Rhoc_d = Debye(F,PHI,MAG,Rref,l,A,w,p,R,RHO,Rho)
elif metodo == 2:
  c = float(input("Qual valor de 'c' para o Método Cole-Cole? \n"))
  F_c,PHI_c,PHIc_c,TK_c,MK_c,RHO_c,Rhoc_c = ColeCole(c,F,PHI,MAG,Rref,l,A,w,p,R,RHO,Rho)
elif metodo == 3:
  c = float(input("Qual valor de 'c' para o Método Cole-Cole? \n"))
  F_d,PHI_d,PHIc_d,TK_d,MK_d,RHO_d,Rhoc_d = Debye(F,PHI,MAG,Rref,l,A,w,p,R,RHO,Rho)
  F_c,PHI_c,PHIc_c,TK_c,MK_c,RHO_c,Rhoc_c = ColeCole(c,F,PHI,MAG,Rref,l,A,w,p,R,RHO,Rho)

if metodo ==1:
  # Grafico 1
  plt.loglog(F_d,-PHI_d*1000,'.k')
  plt.loglog(F_d,-PHIc_d*1000,'k')
  plt.ylabel('phase(mrad)',fontsize = 15)
  plt.xlabel('Frequency(Hz)',fontsize = 15)
  plt.title(nome+ 'Debye', fontsize = 15)
  plt.show()

  # Grafico 2
  plt.figure(figsize=(5,10))
  plt.semilogx(TK_d,MK_d[0],',-k')
  plt.xlabel('Relaxation Time(s) ', fontsize = 15)
  plt.ylabel('Distribuição da Cargabilidade ', fontsize = 15)
  plt.title('Cargabilidade Debye', fontsize = 15)
  plt.plot()
  plt.show()

  # Grafico 3
  plt.loglog(F_d,RHO_d*np.cos(PHI_d),'.k')
  plt.loglog(F_d,np.real(Rhoc_d),'r')
  plt.xlabel('Frequency (Hz) ', fontsize = 15)
  plt.ylabel('\u03C1 (ohm.m) ', fontsize = 15)
  plt.title('Parte Real Debye', fontsize = 15)
  plt.show()

  # Grafico 4
  plt.loglog(F_d,-RHO*np.sin(PHI_d),'.k')
  plt.loglog(F_d,-np.imag(Rhoc_d),'r')
  plt.xlabel('Frequency (Hz)', fontsize = 15)
  plt.ylabel('\u03C1 (ohm.m)', fontsize = 15)
  plt.title('Parte Imaginaria Debye', fontsize = 15)
  plt.show()
elif metodo == 2:
  # Grafico 1
  plt.loglog(F_c,-PHI_c*1000,'.k')
  plt.loglog(F_c,-PHIc_c*1000,'k')
  plt.ylabel('phase(mrad)',fontsize = 15)
  plt.xlabel('Frequency(Hz)',fontsize = 15)
  plt.title(nome+ 'Cole-Cole', fontsize = 15)
  plt.show()

  # Grafico 2
  plt.figure(figsize=(5,10))
  plt.semilogx(TK_c,MK_c[0],',-k')
  plt.xlabel('Relaxation Time(s)', fontsize = 15)
  plt.ylabel('Distribuição da Cargabilidade', fontsize = 15)
  plt.title('Cargabilidade Cole-Cole', fontsize = 15)
  plt.plot()
  plt.show()

  # Grafico 3
  plt.loglog(F_c,RHO_c*np.cos(PHI_c),'.k')
  plt.loglog(F_c,np.real(Rhoc_c),'r')
  plt.xlabel('Frequency (Hz)', fontsize = 15)
  plt.ylabel('\u03C1 (ohm.m)', fontsize = 15)
  plt.title('Parte Real Cole-Cole', fontsize = 15)
  plt.show()

  # Grafico 4
  plt.loglog(F_c,-RHO*np.sin(PHI_c),'.k')
  plt.loglog(F_c,-np.imag(Rhoc_c),'r')
  plt.xlabel('Frequency (Hz)', fontsize = 15)
  plt.ylabel('\u03C1 (ohm.m)', fontsize = 15)
  plt.title('Parte Imaginaria Cole-Cole', fontsize = 15)
  plt.show()
elif metodo == 3 :
  # Grafico 1
  plt.loglog(F_d,-PHI_d*1000,'.k')
  plt.loglog(F_d,-PHIc_d*1000,'k')
  plt.ylabel('phase(mrad)',fontsize = 15)
  plt.xlabel('Frequency(Hz)',fontsize = 15)
  plt.title(nome+ 'Debye', fontsize = 15)
  plt.show()
  plt.loglog(F_c,-PHI_c*1000,'.k')
  plt.loglog(F_c,-PHIc_c*1000,'k')
  plt.ylabel('phase(mrad)',fontsize = 15)
  plt.xlabel('Frequency(Hz)',fontsize = 15)
  plt.title(nome+ 'Cole-Cole', fontsize = 15)
  plt.show()

  # Gráfico 2  
  plt.figure(figsize=(5,10))
  plt.semilogx(TK_d,MK_d[0],',-k')
  plt.xlabel('Relaxation Time(s) ', fontsize = 15)
  plt.ylabel('Distribuição da Cargabilidade ', fontsize = 15)
  plt.title('Cargabilidade Debye', fontsize = 15)
  plt.plot()
  plt.show()
  plt.figure(figsize=(5,10))
  plt.semilogx(TK_c,MK_c[0],',-k')
  plt.xlabel('Relaxation Time(s)', fontsize = 15)
  plt.ylabel('Distribuição da Cargabilidade', fontsize = 15)
  plt.title('Cargabilidade Cole-Cole', fontsize = 15)
  plt.plot()
  plt.show()

  # Grafico 3
  plt.loglog(F_d,RHO_d*np.cos(PHI_d),'.k')
  plt.loglog(F_d,np.real(Rhoc_d),'r')
  plt.xlabel('Frequency (Hz) ', fontsize = 15)
  plt.ylabel('\u03C1 (ohm.m) ', fontsize = 15)
  plt.title('Parte Real Debye', fontsize = 15)
  plt.show()
  plt.loglog(F_c,RHO_c*np.cos(PHI_c),'.k')
  plt.loglog(F_c,np.real(Rhoc_c),'r')
  plt.xlabel('Frequency (Hz)', fontsize = 15)
  plt.ylabel('\u03C1 (ohm.m)', fontsize = 15)
  plt.title('Parte Real Cole-Cole', fontsize = 15)
  plt.show()

  # Grafico 4
  plt.loglog(F_d,-RHO*np.sin(PHI_d),'.k')
  plt.loglog(F_d,-np.imag(Rhoc_d),'r')
  plt.xlabel('Frequency (Hz)', fontsize = 15)
  plt.ylabel('\u03C1 (ohm.m)', fontsize = 15)
  plt.title('Parte Imaginaria Debye', fontsize = 15)
  plt.show()
  plt.loglog(F_c,-RHO*np.sin(PHI_c),'.k')
  plt.loglog(F_c,-np.imag(Rhoc_c),'r')
  plt.xlabel('Frequency (Hz)', fontsize = 15)
  plt.ylabel('\u03C1 (ohm.m)', fontsize = 15)
  plt.title('Parte Imaginaria Cole-Cole', fontsize = 15)
  plt.show()



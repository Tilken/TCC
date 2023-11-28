import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.optimize import nnls
import matplotlib.pyplot as plt

class App:
    def __init__(self, root):
        self.arquivo = None

        self.root = root
        root.title("SIPython")
        self.root.geometry("300x200")

        self.file_path = tk.StringVar()

        # Variáveis para armazenar os valores dos campos
        self.l_value = tk.StringVar()
        self.d_value = tk.StringVar()
        self.Rref_value = tk.StringVar()
        self.c_value = tk.StringVar()

        # Botão para carregar o arquivo
        self.load_button = tk.Button(root, text="Carregar arquivo", command=self.load_file)
        self.load_button.pack()

        # Label para exibir o caminho do arquivo carregado
        self.file_label = tk.Label(root, textvariable=self.file_path)
        self.file_label.pack()

        # Botão para preencher dados das amostras
        self.sample_data_button = tk.Button(root, text="Dados das amostras", command=self.open_sample_data_window)
        self.sample_data_button.pack()

        # Variável para o método de escolha
        self.method = tk.StringVar()

        # Radiobuttons para escolher o método
        self.debye_button = tk.Radiobutton(root, text="Debye", variable=self.method, value="Debye", command=self.choose_method)
        self.debye_button.pack()
        self.cole_cole_button = tk.Radiobutton(root, text="Cole-Cole", variable=self.method, value="Cole-Cole", command=self.choose_method)
        self.cole_cole_button.pack()
        self.both_button = tk.Radiobutton(root, text="Ambos", variable=self.method, value="Ambos", command=self.choose_method)
        self.both_button.pack()

        # Entry para o usuário inserir o valor c
        self.c_label = tk.Label(self.root, text="'c' para Cole-Cole")
        self.c_entry = tk.Entry(root, state='disabled', textvariable=self.c_value)
        self.c_entry.pack()

        # Botão para iniciar o processamento
        self.process_button = tk.Button(root, text="Processar", command=self.start_processing)
        self.process_button.pack()

    def load_file(self):
        file_path = filedialog.askopenfilename()

        try:
            # Suponha que estamos lendo um arquivo csv
            self.arquivo = pd.read_csv(file_path, header=None, sep='\t')
            messagebox.showinfo("Sucesso", "O arquivo foi lido corretamente!")
            self.file_path.set(file_path)
        except Exception as e:
            messagebox.showerror("Erro", f"Ocorreu um erro ao ler o arquivo: {str(e)}")
        return self.arquivo

    def open_sample_data_window(self):
        self.new_window = tk.Toplevel(self.root)
        self.new_window.title("Dados das amostras")
        self.new_window.geometry("300x200")

        # Rótulos e campos para o usuário preencher
        self.l_label = tk.Label(self.new_window, text="l(metros):")
        self.l_label.pack()
        self.l_entry = tk.Entry(self.new_window, textvariable=self.l_value)
        self.l_entry.pack()

        self.d_label = tk.Label(self.new_window, text="d(metros):")
        self.d_label.pack()
        self.d_entry = tk.Entry(self.new_window, textvariable=self.d_value)
        self.d_entry.pack()

        self.Rref_label = tk.Label(self.new_window, text="Rref:")
        self.Rref_label.pack()
        self.Rref_entry = tk.Entry(self.new_window, textvariable=self.Rref_value)
        self.Rref_entry.pack()

        self.Nome_label = tk.Label(self.new_window, text='Nome:')
        self.Nome_label.pack()
        self.Nome_value = tk.StringVar()
        self.Nome_entry = tk.Entry(self.new_window, textvariable=self.Nome_value)
        self.Nome_entry.pack()

        # Botão para confirmar os dados das amostras
        self.confirm_button = tk.Button(self.new_window, text="Confirmar", command=self.new_window.destroy)
        self.confirm_button.pack()

    def choose_method(self):
        if self.method.get() in ["Ambos", "Cole-Cole"]:
            self.c_entry.config(state='normal')
        else:
            self.c_entry.config(state='disabled')

    def start_processing(self):
        method = self.method.get()
        arquivo = self.arquivo
        try:
            l = float(self.l_value.get())
            d = float(self.d_value.get())
            Rref = float(self.Rref_value.get())

            nome = self.Nome_value.get()  # Obter o valor do campo de nome das amostras

            A = np.pi * (d / 2) ** 2
            tK = np.logspace(-4, 3, 700)
            TK = np.transpose(tK)
            n = len(TK)
            H = 10  # Verificar o peso no artigo
            h = 10  # Verificar o peso no artigo

            F = np.array(self.arquivo[0])
            PHI = np.array(self.arquivo[1])
            MAG = np.array(self.arquivo[2])
            w = 2 * np.pi * F  # Equação 4
            p = len(w)
            R = Rref * (10 ** (MAG / 20))  # Equação 5
            RHO = R * A / l  # Equação 6
            Rho = RHO * np.cos(PHI) - (1 * np.array(0 + 1j) * RHO * np.sin(PHI))  # Equação 7
            Fl = np.array(np.logspace(-7, -2, 50))
            FT = np.concatenate((F, Fl))
            r = interpolate.interp1d(F, R, kind='linear', fill_value='extrapolate')
            resultado_inter = r(FT)
            rho = resultado_inter * A / l
            DC = rho[-1]
            RHO_N_RE = abs((np.real(Rho) - DC)) / DC
            RHO_N_IM = (np.imag(Rho)) / DC
            G1 = np.zeros((p,n))
            G2 = np.zeros((p,n))
            
            def Debye(F, PHI, MAG, Rref, l, A, w, p, R, RHO, Rho, RHO_N_IM, RHO_N_RE, DC):
                    c=1
                    for i in range(p):
                        for j in range(n):
                            g1 = (w[i]*TK[j])**c
                            g2 = w[i]*TK[j]
                            G1[i,j] = g1/(1+g1)
                            G2[i,j] = g2/(1+g1)

                    mG2 = nnls(G2,RHO_N_IM)
                    mG1 = nnls(G1,RHO_N_RE)
                    soma_mG1 = np.sum(mG1[0])
                    soma_mG2 = np.sum(mG2[0]) # Diferença no vetor

                # Passo 4  Obtendo RHO_0

                    RHO_0 = DC * (1+soma_mG2 - soma_mG1) # Diferença no Rho_0 por conta do vetor mG2

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
                    return F, PHI, PHIc, TK, MK, RHO, Rhoc


            def ColeCole(c, F, PHI, MAG, Rref, l, A, w, p, R, RHO, Rho, RHO_N_IM, RHO_N_RE, DC):
                    for i in range(p):
                        for j in range(n):
                            g1 = (w[i]*TK[j])**c
                            g2 = w[i]*TK[j]
                            G1[i,j] = g1/(1+g1)
                            G2[i,j] = g2/(1+g1)

                    mG2 = nnls(G2,RHO_N_IM)
                    mG1 = nnls(G1,RHO_N_RE)
                    soma_mG1 = np.sum(mG1[0])
                    soma_mG2 = np.sum(mG2[0]) # Diferença no vetor

                # Passo 4  Obtendo RHO_0

                    RHO_0 = DC * (1+soma_mG2 - soma_mG1) # Diferença no Rho_0 por conta do vetor mG2

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
                    return F, PHI, PHIc, TK, MK, RHO, Rhoc
            
            if method == "Ambos":
                c = float(self.c_value.get())
                F_d, PHI_d, PHIc_d, TK_d, MK_d, RHO_d, Rhoc_d = Debye(F, PHI, MAG, Rref, l, A, w, p, R, RHO, Rho,
                                                                     RHO_N_IM, RHO_N_RE, DC)
                F_c, PHI_c, PHIc_c, TK_c, MK_c, RHO_c, Rhoc_c = ColeCole(c, F, PHI, MAG, Rref, l, A, w, p, R, RHO, Rho,
                                                                        RHO_N_IM, RHO_N_RE, DC)
                # Plots Comparativos
                plt.loglog(F_d, -PHI_d * 1000, '.k', label='Dados')
                plt.loglog(F_d, -PHIc_d * 1000, 'k', label='Debye - Ajuste')
                plt.loglog(F_c, -PHIc_c * 1000, 'r', label='Cole-Cole - Ajuste')
                plt.ylabel('phase(mrad)', fontsize=15)
                plt.xlabel('Frequency(Hz)', fontsize=15)
                plt.title(nome + ' Debye vs Cole-Cole', fontsize=15)
                # Adicionando legenda
                plt.legend()
                plt.show()
                
                
                #Plot das fases de ambos
                fig, axs = plt.subplots(1, 2, figsize=(12, 12))
                
                axs[0].loglog(F_d, -PHI_d * 1000, '.k')
                axs[0].loglog(F_d, -PHIc_d * 1000, 'k')
                axs[0].set_ylabel('phase(mrad)', fontsize=15)
                axs[0].set_xlabel('Frequency(Hz)', fontsize=15)
                axs[0].set_title(nome + ' Debye', fontsize=15)
                     
                axs[1].loglog(F_c, -PHI_c * 1000, '.k')
                axs[1].loglog(F_c, -PHIc_c * 1000, 'k')
                axs[1].set_ylabel('phase(mrad)', fontsize=15)
                axs[1].set_xlabel('Frequency(Hz)', fontsize=15)
                axs[1].set_title(nome + ' Cole-Cole', fontsize=15)  
                                
                plt.show()
                
                #plot Cargabilidade de ambos
                fig, axs = plt.subplots(1, 2, figsize=(12, 12))

                axs[0].semilogx(TK_d, MK_d[0], ',-k')
                axs[0].set_xlabel('Relaxation Time(s)', fontsize=12)
                axs[0].set_ylabel('Distribuição da Cargabilidade', fontsize=12)
                axs[0].set_title('Cargabilidade Debye', fontsize=12)
                
                axs[1].semilogx(TK_c, MK_c[0], ',-k')
                axs[1].set_xlabel('Relaxation Time(s)', fontsize=12)
                axs[1].set_ylabel('Distribuição da Cargabilidade', fontsize=12)
                axs[1].set_title('Cargabilidade Cole-Cole', fontsize=12)
                

                plt.show()
                
                
                #Plot parte real de ambos
                fig, axs = plt.subplots(1, 2, figsize=(12, 12))
                
                axs[0].loglog(F_d, RHO_d * np.cos(PHI_d), '.k')
                axs[0].loglog(F_d, np.real(Rhoc_d), 'r')
                axs[0].set_xlabel('Frequency (Hz) ', fontsize=15)
                axs[0].set_ylabel('\u03C1 (ohm.m) ', fontsize=15)
                axs[0].set_title('Parte Real Debye', fontsize=15)
                
                axs[1].loglog(F_c, RHO_c * np.cos(PHI_c), '.k')
                axs[1].loglog(F_c, np.real(Rhoc_c), 'r')
                axs[1].set_xlabel('Frequency (Hz)', fontsize=15)
                axs[1].set_ylabel('\u03C1 (ohm.m)', fontsize=15)
                axs[1].set_title('Parte Real Cole-Cole', fontsize=15)
                
                plt.show()
                
                #plot parte Imaginária de ambos
                fig, axs = plt.subplots(1, 2, figsize=(12, 12))
                
                axs[0].loglog(F_d, -RHO * np.sin(PHI_d), '.k')
                axs[0].loglog(F_d, -np.imag(Rhoc_d), 'r')
                axs[0].set_xlabel('Frequency (Hz)', fontsize=15)
                axs[0].set_ylabel('\u03C1 (ohm.m)', fontsize=15)
                axs[0].set_title('Parte Imaginaria Debye', fontsize=15)
                
                axs[1].loglog(F_c, -RHO * np.sin(PHI_c), '.k')
                axs[1].loglog(F_c, -np.imag(Rhoc_c), 'r')
                axs[1].set_xlabel('Frequency (Hz)', fontsize=15)
                axs[1].set_ylabel('\u03C1 (ohm.m)', fontsize=15)
                axs[1].set_title('Parte Imaginaria Cole-Cole', fontsize=15)
                
                plt.show()

            elif method == "Debye":
                F_d, PHI_d, PHIc_d, TK_d, MK_d, RHO_d, Rhoc_d = Debye(F, PHI, MAG, Rref, l, A, w, p, R, RHO, Rho,
                                                                     RHO_N_IM, RHO_N_RE, DC)

                # Plots para o método Debye
                plt.loglog(F_d, -PHI_d * 1000, '.k')
                plt.loglog(F_d, -PHIc_d * 1000, 'k')
                plt.ylabel('phase(mrad)', fontsize=15)
                plt.xlabel('Frequency(Hz)', fontsize=15)
                plt.title(nome + ' Debye', fontsize=15)
                plt.show()

                plt.figure(figsize=(5, 10))
                plt.semilogx(TK_d, MK_d[0], ',-k')
                plt.xlabel('Relaxation Time(s) ', fontsize=15)
                plt.ylabel('Distribuição da Cargabilidade ', fontsize=15)
                plt.title('Cargabilidade Debye', fontsize=15)
                plt.plot()
                plt.show()

                plt.loglog(F_d, RHO_d * np.cos(PHI_d), '.k')
                plt.loglog(F_d, np.real(Rhoc_d), 'r')
                plt.xlabel('Frequency (Hz) ', fontsize=15)
                plt.ylabel('\u03C1 (ohm.m) ', fontsize=15)
                plt.title('Parte Real Debye', fontsize=15)
                plt.show()

                plt.loglog(F_d, -RHO * np.sin(PHI_d), '.k')
                plt.loglog(F_d, -np.imag(Rhoc_d), 'r')
                plt.xlabel('Frequency (Hz)', fontsize=15)
                plt.ylabel('\u03C1 (ohm.m)', fontsize=15)
                plt.title('Parte Imaginaria Debye', fontsize=15)
                plt.show()

            elif method == "Cole-Cole":
                c = float(self.c_value.get())
                F_c, PHI_c, PHIc_c, TK_c, MK_c, RHO_c, Rhoc_c = ColeCole(c, F, PHI, MAG, Rref, l, A, w, p, R, RHO, Rho,
                                                                        RHO_N_IM, RHO_N_RE, DC)

                # Plots para o método Cole-Cole
                plt.loglog(F_c, -PHI_c * 1000, '.k')
                plt.loglog(F_c, -PHIc_c * 1000, 'k')
                plt.ylabel('phase(mrad)', fontsize=15)
                plt.xlabel('Frequency(Hz)', fontsize=15)
                plt.title(nome + ' Cole-Cole', fontsize=15)
                plt.show()

                plt.figure(figsize=(5, 10))
                plt.semilogx(TK_c, MK_c[0], ',-k')
                plt.xlabel('Relaxation Time(s)', fontsize=15)
                plt.ylabel('Distribuição da Cargabilidade', fontsize=15)
                plt.title('Cargabilidade Cole-Cole', fontsize=15)
                plt.plot()
                plt.show()

                plt.loglog(F_c, RHO_c * np.cos(PHI_c), '.k')
                plt.loglog(F_c, np.real(Rhoc_c), 'r')
                plt.xlabel('Frequency (Hz)', fontsize=15)
                plt.ylabel('\u03C1 (ohm.m)', fontsize=15)
                plt.title('Parte Real Cole-Cole', fontsize=15)
                plt.show()

                plt.loglog(F_c, -RHO * np.sin(PHI_c), '.k')
                plt.loglog(F_c, -np.imag(Rhoc_c), 'r')
                plt.xlabel('Frequency (Hz)', fontsize=15)
                plt.ylabel('\u03C1 (ohm.m)', fontsize=15)
                plt.title('Parte Imaginaria Cole-Cole', fontsize=15)
                plt.show()

        except Exception as e:
            messagebox.showerror("Erro", f"Ocorreu um erro ao processar os dados: {str(e)}")

root = tk.Tk()
app = App(root)
root.mainloop()

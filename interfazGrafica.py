# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 20:24:58 2021

@author: edgar

Interfaz grafica para detectar neumonia en radiografías de rayos-X
"""

import os
import cv2
import pickle
import tkinter as tk
import numpy as np
from tkinter import filedialog
from PIL import ImageTk, Image
import Clasificador_incremental_no_add_prototypes_v11 as cinap

__version__ = '2.5'


# %% Cargamos datos de entrenamiento y prueba
# Prototipos
f = open('prototiposOVOnvpk15.pkl','rb')
Prototipos = pickle.load(f)
# Eigenfaces seleccionados
f = open('uinSelOVOnvpk15.pkl','rb')
uinEigen = pickle.load(f)
# Imagen promedio
f = open('psiOVOnvpk15.pkl','rb')
psi = pickle.load(f)
# Desviacion estandar del conjunto de entrenamiento
f = open('desvStdOVOnvpk15.pkl','rb')
desvStd = pickle.load(f)
# Promedio del conjunto de entrenamiento
f = open('promOVOnvpk15.pkl','rb')
prom = pickle.load(f)

# Valor de k para kNN
k = 15

class interfazGrafica():
    
    def __init__(self,uinEigen,psi,desvStd,prom,Prototipos,k):
        # Variables necesarias
        self.uinEigen = uinEigen
        self.psi = psi
        self.desvStd = desvStd
        self.prom = prom
        self.Prototipos = Prototipos
        self.K = k
        self.imLoaded = False
        self.yaProcesado = False
        self.primero = True
        self.tempdir = None
        
        # Creamos la raiz
        self.raiz = tk.Tk()
        # self.raiz.attributes('-fullscreen', True) 
        self.raiz.geometry('1050x650')
        self.raiz['bg'] = '#002851'
        self.raiz.title('Pneumonia Detection')
        self.raiz.iconbitmap("ico.ico")
        
        # Mostramos imagen de bienvenida
        self.canvas = tk.Canvas(width=650, height=650, bg='black')
        pilImage = Image.open("Illustration.jpg")
        # Redimensionamos la imagen
        [ancho,alto] = pilImage.size
        imRedim = pilImage.resize([650,650])
        # Empaquetamos la imagen
        self.image = ImageTk.PhotoImage(imRedim)
        self.canvas.grid(column=0,row=0)
        self.canvasImage = self.canvas.create_image(0,0,image=self.image,
                                                    anchor=tk.NW)
        self.canvas.itemconfig(self.canvasImage,image=self.image)    
        
        # Seleccion seccion de imagen
        self.canvas.bind("<ButtonPress-1>",self.mouseIzqPres)
        self.canvas.bind("<B1-Motion>",self.dibujar)
        self.canvas.bind("<Double-1>",self.limpiar)
        self.canvas.bind("<ButtonRelease-1>",self.mouseIzqSolt)
        self.raiz.bind("<KeyPress>",self.mover)
        self.recorte = False
        self.dibujo = None
        
        # Frame
        self.miFrame = tk.Frame()
        self.miFrame.grid(column=1,row=0)
        # Ajuste automatico al tamaño de la ventana
        tk.Grid.columnconfigure(self.raiz, 1, weight=1)
        tk.Grid.rowconfigure(self.raiz, 1, weight=1)
        # self.miFrame.config(bg='white')
        self.miFrame['bg'] = '#002851'
        # Etiquetas de instrucciones
        etiqInst = tk.Label(self.miFrame,
                              text="Seleccione la imagen a procesar")
        etiqInst.grid(column=0,row=0,columnspan=3)
        etiqInst.config(bg="#002851",fg="#D1290B",justify="center",
                             font=("Cambria",16,"bold"))
        etiqInst2 = tk.Label(self.miFrame,
                              text="desde Abrir en el menu archivo")
        etiqInst2.grid(column=0,row=1,columnspan=3)
        etiqInst2.config(bg="#002851",fg="#D1290B",justify="center",
                              font=("Cambria",16,"bold"))
        etiqInst3 = tk.Label(self.miFrame,
                              text="despues recorte la imagen con el mouse")
        etiqInst3.grid(column=0,row=2,columnspan=3,pady=(0,10))
        etiqInst3.config(bg="#002851",fg="#D1290B",justify="center",
                              font=("Cambria",16,"bold"))
        etiqInst4 = tk.Label(self.miFrame,
                              text="Consulte la ayuda en el menu ayuda")
        etiqInst4.grid(column=0,row=3,columnspan=3,pady=(0,70))
        etiqInst4.config(bg="#002851",fg="white",justify="center",
                              font=("Cambria",15))
        
        # Etiquetas mensaje de accuaracy
        etiqAcc = tk.Label(self.miFrame,
                           text="La tasa de reconocimiento")
        etiqAcc.grid(column=0,row=4,columnspan=3)
        etiqAcc.config(bg="#002851",fg="gray",justify="center",
                             font=("Cambria",15))
        etiqAcc2 = tk.Label(self.miFrame,
                           text="es de alrededor del: 92 %")
        etiqAcc2.grid(column=0,row=5,columnspan=3)
        etiqAcc2.config(bg="#002851",fg="gray",justify="center",
                             font=("Cambria",15))
        # Boton para procesar
        self.botPro = tk.Button(self.miFrame,text='Procesar',
                           command=self.procesarBot,font=('TkDefaultFont',12))
        self.botPro.grid(column=1,row=6,pady=(100,20),ipadx=5)
        self.botPro["state"] = "disabled"
        # Etiqueta Resultado
        etiqRes = tk.Label(self.miFrame,text="Probabilidad de neumonia")
        etiqRes.grid(column=1,row=7)
        etiqRes.config(bg="#002851",fg="white",justify="center",
                            font=("Cambria",15))
        # Pantalla de RESULTADO
        self.resultado = tk.StringVar()
        self.pantalla = tk.Entry(self.miFrame,textvariable=self.resultado)
        self.pantalla.grid(column=1,row=8)
        self.pantalla.config(bg="white",fg="#002851",justify="center",
                             font=('TkDefaultFont', 15))
        #Etiqueta version
        etiqVers = tk.Label(self.miFrame,text="V "+__version__,)
        etiqVers.grid(column=0,row=9,pady=(120,0),padx=20)
        # Boton salir
        botsalir = tk.Button(self.miFrame, text='Salir', 
                                    command=self.raiz.destroy)
        botsalir.grid(column=2,row=9,pady=(120,0),ipadx=15)

        # Creamos barra menu
        barramenu = tk.Menu(self.raiz)
        self.raiz.config(menu=barramenu)
        # Creamos los menus
        filemenu = tk.Menu(barramenu, tearoff=0)
        # editmenu = tk.Menu(barramenu, tearoff=0)
        helpmenu = tk.Menu(barramenu, tearoff=0)
        # Agregamos los menus a la barra
        barramenu.add_cascade(label="Archivo", menu=filemenu)
        # barramenu.add_cascade(label="Editar", menu=editmenu)
        barramenu.add_cascade(label="Ayuda", menu=helpmenu)
        # Agregamos opciones menu archivo
        filemenu.add_command(label="Abrir", command=self.cargarImagen)
        filemenu.add_command(label="Guardar", command=self.guardarImagen)
        filemenu.add_separator()
        filemenu.add_command(label="Salir", command=self.raiz.destroy)
        # Agregamos opciones al menu ayuda
        helpmenu.add_command(label="Ayuda", command=self.ayuda)
        helpmenu.add_command(label="Ejemplo de recorte", command=self.ejemploSel)
        helpmenu.add_separator()
        helpmenu.add_command(label="Acerca de...", command=self.acercaDe)
        
        # Ciclo infinito
        self.raiz.mainloop()
        
    def cargarImagen(self):
        self.botPro["state"] = "disabled"
        if self.primero == True:
            archivo = filedialog.askopenfilename(initialdir="/", 
                                                  title="Seleccione imagen", 
                                                  filetypes=(("all files", ".*"),
                                                             ("jpeg files", "*.jpeg"),
                                                             ("jpg files", "*.jpg"),
                                                             ("png files", "*.png")
                                                             ))
            path, file = os.path.split(archivo)
            self.tempdir = path
            self.primero = False
        else:
            if self.tempdir is not None:
                archivo = filedialog.askopenfilename(initialdir=self.tempdir, 
                                                  title="Seleccione una imagen", 
                                                  filetypes=(("all files", ".*"),
                                                             ("jpeg files", "*.jpeg"),
                                                             ("jpg files", "*.jpg"),
                                                             ("png files", "*.png")
                                                             ))
                path, file = os.path.split(archivo)
                self.tempdir = path
            else:
                archivo = filedialog.askopenfilename(initialdir="/", 
                                                      title="Seleccione imagen", 
                                                      filetypes=(("all files", ".*"),
                                                                 ("jpeg files", "*.jpeg"),
                                                                 ("jpg files", "*.jpg"),
                                                                 ("png files", "*.png")
                                                                 ))
                path, file = os.path.split(archivo)
                self.tempdir = path
                self.primero = False
                    
        # Limpio pantalla
        self.resultado.set("")
        # Fondo de pantalla en blanco
        self.pantalla.config(bg="white")
        # Mascara
        plantilla = np.zeros((650,650),dtype='uint8')
        # Si se cargo bien la ruta
        if len(archivo) > 0:
            self.yaProcesado = False
            # Cargamos imagen
            imEnt = cv2.imread(archivo)
            # Variable para saber si ya se cargó una imagen
            self.imLoaded = True
            # Dimensiones de la imagen
            [alto,ancho,x] = imEnt.shape
            # Imagen en escala de grises
            im = cv2.cvtColor(imEnt, cv2.COLOR_RGB2GRAY) #convierte a gris
            # Redimensionamos para mostrar en interfaz
            if ancho > alto:
                newAncho = 650
                newAlto = int((alto*650)/ancho)
            else:
                newAncho = int((ancho*650)/alto)
                newAlto = 650
            imPIL = Image.fromarray(im)
            imRedim = imPIL.resize([newAncho,newAlto])
            # Im array
            imRedim2 = np.asarray(imRedim)
            # Ponemos dentro de mascara
            if newAlto or newAncho != 650:
                # Si es mas ancha
                if newAncho == 650:
                    dify = 650 - newAlto
                    iniy = int(dify/2)
                    plantilla[iniy:iniy+newAlto,:] = imRedim2 
                # Si es mas alta
                if newAlto == 650:
                    difx = 650 - newAncho
                    inix = int(difx/2)
                    plantilla[:,inix:inix+newAncho] = imRedim2
            
            # Imagen con mascara
            self.imIni = plantilla
            imPIL = Image.fromarray(plantilla)
            # Empaquetamos
            self.image = ImageTk.PhotoImage(imPIL)
            self.canvas.itemconfig(self.canvasImage,image=self.image)
            # actualizamos
            self.raiz.focus()
            self.raiz.update()
        
    def guardarImagen(self):
        if self.imLoaded == True:
            archivoG = filedialog.asksaveasfilename(initialdir="/",
                                                    title="Guardar archivo",
                                                    defaultextension=".jpg",
                                                    filetypes=(("jpeg files", "*.jpeg"),
                                                             ("jpg files", "*.jpg"),
                                                             ("png files", "*.png")
                                                             ))
            cv2.imwrite(archivoG,self.imIni)
        else:
            self.resultado.set("Cargue una imagen")
            self.pantalla.config(bg='#F55713')
        
    def procesarBot(self):
        # Revisamos si ya se cargó una imagen
        if (self.imLoaded == False) or (self.recorte == False) or (self.yaProcesado == True):
            if (self.imLoaded == False):    
                self.resultado.set("Cargue una imagen")
                self.pantalla.config(bg='#F55713')
            elif (self.yaProcesado == True):    
                self.resultado.set("Cargue nueva imagen")
                self.pantalla.config(bg='#F55713')
            else:
                self.resultado.set("Recorte la imagen")
                self.pantalla.config(bg='#F55713')
        else:
            self.yaProcesado = True
            # Dimension original
            [alto,ancho] = self.imIni.shape
            # Redimensionamos para mostrar en interfaz
            if ancho > alto:
                newAncho = 650
                newAlto = int((alto*650)/ancho)
            else:
                newAncho = int((ancho*650)/alto)
                newAlto = 650
            # Imagen PIL
            imPIL = Image.fromarray(self.imIni)
            # Redimension 650x650
            imRedim = imPIL.resize([newAncho,newAlto])
            # Conversion
            imRec = np.asarray(imRedim)
            # Recortamos seleccion usario
            self.imRec = imRec[self.origeny:self.finy,self.origenx:self.finx]
            # Mejora de contraste
            self.ecuHist = cv2.equalizeHist(self.imRec)
            # dimension del recorte al area de la ventana
            self.imRecV = cv2.resize(self.ecuHist,(650,650))
            imPIL = Image.fromarray(self.imRecV)
            self.image = ImageTk.PhotoImage(imPIL)
            self.canvas.itemconfig(self.canvasImage,image=self.image)
            self.canvas.delete(self.dibujo) 
            [nAlto,nAncho] = self.ecuHist.shape
            if nAncho != 256 or nAlto != 256:
                imGrayRedim = cv2.resize(self.ecuHist,(256,256))
            else:
                imGrayRedim = self.ecuHist
            # Convertimos a arreglo de numpy
            imVec = np.array(imGrayRedim)
            imVec = imVec.reshape((-1))
            # print(imVec.shape)
            proba = self.procesar(imVec)
            
    def procesar(self, imVec):
        # Proyeccion
        imProy = cinap.proy1Tst(imVec, self.uinEigen, self.psi)
        # Normalizamos imagen de entrada
        imNorm = cinap.normStdTst(imProy, self.desvStd, self.prom)
        imNormSel = imNorm
        # Valores para indexar mas legiblemente las listas 
        Feat = 0  # Características
        Clas = 1  # Clase
        # Obtenemos la dimensión de características
        data_dim = imNormSel.shape[0]  
        # Prototipos
        Prototypes_tst = cinap.listofarray2array(self.Prototipos, Feat ) 
        # Etiquetas
        Prototypes_labels_tst = cinap.list2array(self.Prototipos, Clas)  
        # knn
        eL,n1,n0,CLS,v0,v1 = cinap.KNNvot(imNormSel.reshape(data_dim,1), 
                                          Prototypes_tst, Prototypes_labels_tst,
                                          self.K)  
        if eL == 0:
            self.pantalla.config(bg='#4EE510')
            self.resultado.set("Normal")
        else:
            self.pantalla.config(bg='#E53010')
            prob = round((v1 / (v0+v1)) * 100,2)
            self.resultado.set(str(prob)+"%")
        return(eL)
    
    
    # SELECCION ZONA DE IMAGEN
    # evento al presionar boton izquierdo
    def mouseIzqPres(self, evento):
        self.recorte = False
        self.origenx = evento.x
        self.origeny = evento.y
    # evento al mover el mouse    
    def dibujar(self, evento):
        if self.dibujo:
            self.canvas.delete(self.dibujo)
        self.dibujo = self.canvas.create_rectangle(self.origenx,self.origeny,
                                              (evento.y-self.origeny)+self.origenx,
                                              evento.y,outline='red',width=4)
    # evento al hacer doble click izquierdo
    def limpiar(self, evento):
        evento.widget.delete(self.dibujo)
        
    # evento al soltar boton izquierdo
    def mouseIzqSolt(self, evento):
        if self.dibujo:
            self.finy = evento.y
            dif = self.finy - self.origeny
            self.finx = self.origenx + dif
            if (self.origenx != self.finx) and (self.origeny != self.finy):
                self.recorte = True
                self.botPro["state"] = "normal"

    def mover(self,evento):
        if self.recorte:
            if evento.keysym=='Right':
                self.canvas.move(self.dibujo, 3, 0)
                self.origenx = self.origenx + 3
                self.finx = self.finx + 3
            if evento.keysym=='Left':
                self.canvas.move(self.dibujo, -3, 0)
                self.origenx = self.origenx - 3
                self.finx = self.finx - 3
            if evento.keysym=='Down':
                self.canvas.move(self.dibujo, 0, 3)
                self.origeny = self.origeny + 3
                self.finy = self.finy + 3
            if evento.keysym=='Up':
                self.canvas.move(self.dibujo, 0, -3)
                self.origeny = self.origeny - 3
                self.finy = self.finy - 3
            
    # MENUS
    def acercaDe(self):
        """
        Nueva ventana que muestra el contenido acerca de....

        Returns
        -------
        None.

        """
        # Ventana
        acerca = tk.Toplevel()
        acerca.geometry("300x350")
        acerca.resizable(width=False, height=False)
        acerca.title("Acerca de")
        miframe2 = tk.Frame(acerca,relief=tk.RAISED)
        miframe2.pack(expand=True, fill=tk.BOTH)
        miframe2['bg'] = '#002851'
        # Imagen en ventana
        ima = tk.PhotoImage(file="NORMALgif.gif")
        logo = tk.Label(miframe2, image=ima)
        logo.pack(side=tk.LEFT,pady=(20,275))
        # Version
        etiq = tk.Label(miframe2, text="Pneumonia Detection V."+__version__, 
                     foreground='blue')
        etiq.pack()

        texto = "Creado por Edgar I. T. M. como parte del \n"\
                "trabajo de tesis: Sistema de Aprendizaje \n"\
                "Automatico para la deteccion de deumonia.\n"\
                "Asesores: M.C. J. Francisco Portillo Robledo \n"\
                "Dr. Eugenio Salvador Ayala Raggi.\n\n"\
                "Este proyecto es parte del trabajo realizado\n"\
                "durante la maestria para proporcionar una \n"\
                "herramienta a los radiologos para detectar \n"\
                "neumonia en radiografias de torax. \n\n"\
                "Versiones de herramientas de desarrollo:\n"\
                "python: 3.7.6 \n"\
                "tkinter: 8.6 \n"\
                "spyder: 4.0.1 \n\n"\
                "Para mayor informacin enviar correo a: \n"\
                "salvador.raggi@correo.buap.mx "
        etiq2 = tk.Label(miframe2, justify=tk.LEFT, text=texto)
        etiq2.pack(after=etiq)
        etiq2.config(bg="#002851",fg="white")
        # Boton para salir
        boton1 = tk.Button(miframe2, text="Aceptar", command=acerca.destroy)
        boton1.pack(after=etiq2,pady=(15,0))
        
        boton1.focus_set()
        acerca.transient(self.raiz)
        self.raiz.wait_window(acerca)
        
    def ayuda(self):
        """
        Ventana de ayuda

        Returns
        -------
        None.

        """
        # Nueva ventana
        mAyuda = tk.Toplevel()
        mAyuda.geometry("700x600")
        mAyuda.resizable(width=False,height=False)
        mAyuda.title("Ayuda")
        # Utilizamos canvas
        base = tk.Canvas(mAyuda, width=700,height=600,bg='#232629')
        base.pack()
        
        base.create_text(10,20,fill="white",text="COMENZANDO", anchor=tk.SW)
        # Linea blanca
        base.create_line(0,25,700,25,fill="white")
        # Texto a desplegar
        texto1 = "Para iniciar abra el menu archivo ->abrir, a continuacion "\
        "se abrira una ventana del explorador en la que se puede seleccionar "\
        "una imagen en formato .jpeg, .jpg o .png. Primero seleccione la "\
        "extension de la imagen, de otra manera no se mostraran imagenes en "\
        "el explorador de archivos. Despues de seleccionar la extension y la "\
        "imagen deseada, dar click en abrir y la imagen seleccionada se"\
        "mostrara en la interfaz. Si la imagen que se muestra es la que usted "\
        "desea diagnosticar, el siguiente paso es recortar la zona de "\
        "los pulmones. Para esto, presione el boton izquierdo del mouse y "\
        "mantengalo presionado para seleccionar la zona deseada. Si desea "\
        "volver a realizar la selección, oprima el botón izquierdo dos veces "\
        "y la seleccion actual se borrara perimtiendo hacer una nueva "\
        "seleccion. Tambien pude ajustar la posición del cuadrado con las "\
        "teclas izquierda, derecha, arriba y abajo del teclado. Una vez "\
        "seleccionada la imagen de click en el boton de procesar y al "\
        "finalizar el analisis, se mostrara la probabilidad de neumonia "\
        "con el fondo en color rojo o si esta en un estado normal con el "\
        " fondo en color verde."
        base.create_text(40,40,fill="white",text=texto1,anchor=tk.NW,width=600)
        # Linea blanca
        base.create_line(0,220,700,220,fill="white")
        # Texto salir
        base.create_text(10,225,fill="white",text="SALIR", anchor=tk.NW)
        # Linea blanca
        base.create_line(0,240,700,240,fill="white")
        # Texto a desplegar
        texto2 = "Para salir se tienen 3 opciones, cerrar la ventana desde "\
        "cerrar en la esquina superior derecha, dar click sobre el boton "\
        "salir situado en la esquina inferior derecha o desde el menu archivo"\
        " haciendo click en salir."
        base.create_text(40,250,fill="white",text=texto2,anchor=tk.NW,width=600)
        # Linea blanca
        base.create_line(0,300,700,300, fill="white")
        
    def ejemploSel(self):
        # Nueva ventana
        mEjemploSel = tk.Toplevel()
        # Tamaño de la ventana
        mEjemploSel.geometry("1200x420")
        # No se permite redimensionar
        mEjemploSel.resizable(width=False,height=False)
        # Titulo de la ventana
        mEjemploSel.title("Ejemplo de seleccion de zona de pulmones")
        # Creamos un linzo de canvas
        base2 = tk.Canvas(mEjemploSel,width=1200,height=420,bg='#232629')
        base2.pack()
        # Imagenes a cargar
        pilImage1 = Image.open("ejemploRecorte.png")
        pilImage2 = Image.open("ejemploRecorte2.png")
        pilImage3 = Image.open("ejemploRecorte3.png")
        # Redimensionamos las imagenes
        imRedim1 = pilImage1.resize([400,400])
        imRedim2 = pilImage2.resize([400,400])
        imRedim3 = pilImage3.resize([400,400])
        # Creamos imagen compatibles con tk
        image1 = ImageTk.PhotoImage(imRedim1)
        image2 = ImageTk.PhotoImage(imRedim2)
        image3 = ImageTk.PhotoImage(imRedim3)
        # Colocamos en pantalla
        base2.create_image(200,200,image=image1)
        base2.create_image(600,200,image=image2)
        base2.create_image(1000,200,image=image3)
        # Loop para poder visualizar las imagenes
        mEjemploSel.mainloop()
        


if __name__ == '__main__':
    gui = interfazGrafica(uinEigen,psi,desvStd,prom,Prototipos,k)

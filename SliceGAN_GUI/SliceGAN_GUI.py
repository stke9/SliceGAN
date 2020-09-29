from tkinter import  ttk, filedialog
from tkinter import *
from tkinter import font
from PIL import Image, ImageTk, ImageOps
from SliceGAN_util import Train, Architect
import os
import numpy as np
import torch
import torch.nn as nn
import threading

## Window class
class Window(Frame):
    def __init__(self, master = None):
        Frame.__init__(self, master)
        self.master = master
        self.init_window()

    #initialise Window
    def init_window(self):
        #Add title and basics
        self.master.title('SliceGAN')
        self.nb = ttk.Notebook(self.master)
        self.nb.pack(side = TOP, fill = BOTH, expand = 1)
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)
        self.nz = 32

        ##Init home frame
        self.init_f1()
        self.nb.add(self.f1, text = 'Home')

    def init_f1(self):
        # initialise variables and buttons in frame 1
        self.f1 = Frame(self.nb)

        self.pro = 0
        self.training_image_size = 64
        self.channels = 2
        self.imgs = []
        self.statuslab = Label(self.f1, text='status: Image Loading')
        self.statuslab.grid(column=0, row=7, columnspan=3, sticky='sw')
        self.resize_with_window()
        self.cube_loaded = False
        self.advanced = Checkbutton(self.f1, text='Pro Mode', onvalue=1, offvalue=0, command=self.turn_pro,
                                    variable=self.pro)
        self.advanced.grid(column=0, row=0)
        self.imsize = IntVar(self.f1)
        self.imsize.set(64)
        self.set_imsize(None)
        lab = Label(self.f1, text = 'Training Image\n size:')
        lab.grid(column = 0, row = 1)
        self.sizeEntry =OptionMenu(self.f1,self.imsize,64,128,256, command = self.set_imsize)
        self.sizeEntry.grid(column=0, row=2)
        self.finishbut = Button(self.f1, text='Train', command=self.start)
        self.finishbut.grid(column=7, row=7, sticky='sw')
        self.f1.grid_rowconfigure(7, weight=1)
        # Initiate Image instances
        self.imgs.append(self.Img(0, 'red', self))

    def init_f2(self):
        ## Initialise variables and buttons Architecture frame

        self.f2 = Frame(self.nb)
        self.master.bind('<Return>', self.update_nets)
        self.nb.add(self.f2, text='Architecture')
        self.nb.select(self.f1)

        self.nets = [self.NetImg(1,'GeNeRaToR',self), self.NetImg(8,'DiScRiMiNaToR',self)]
        self.master.update()
        self.update_nets()

        ttk.Separator(self.f2, orient=HORIZONTAL).grid(column=0, row=6, rowspan=6)

    def update_nets(self, event = None):
        self.net_imwidth = max(self.nets[0].lay_num, self.nets[1].lay_num)
        for net in self.nets:
            net.update_arch()

    def set_imsize(self, event):
        self.training_image_size = self.imsize.get()
        self.auto_params(True)
        self.update_images()
        if self.pro:
            self.f2.grid_remove()
            self.f2.destroy()
            self.init_f2()

    def turn_pro(self):
        if self.pro:
            self.pro = 0
            self.f1.grid_remove()
            self.f1.destroy()
            self.f2.grid_remove()
            self.f2.destroy()
            self.init_f1()
            self.resize_with_window()

            self.update_images()
            self.nb.add(self.f1, text='Image Load')
            self.nb.select(self.f1)

        else:
            self.auto_params(True)
            self.pro = 1
            self.resize_with_window()

            self.init_f2()
            for column, colour in zip(range(1,3), ['green', 'blue']):
                self.imgs.append(self.Img(column, colour, self))
            self.imgs[0].rotlft = Button(self.f1, text='↻', command=lambda dir='ac': self.imgs[0].rot_image(dir))
            self.imgs[0].rotlft.grid(column=1, row=2, padx=(0, 25))
            self.imgs[0].rotrt = Button(self.f1, text='↺', command=lambda dir='c': self.imgs[0].rot_image(dir))
            self.imgs[0].rotrt.grid(column= 1, row=2, padx=(80, 0))
            self.update_images()

    def resize_with_window(self):
        if self.pro:
            self.master.geometry('900x600')
            self.column_width = 200
        else:
            self.master.geometry('500x600')
            self.column_width = 250

    def update_images(self):
        for img in self.imgs:
            img.update()

    def auto_params(self, auto):
        self.nz = 32
        if auto:
            n_lay_auto = 5 + self.training_image_size // 128
            #define strides, paddings and kernals
            self.gkernals = [4]*n_lay_auto
            self.dkernals =  [4]*n_lay_auto
            self.gstrides =  [2]*n_lay_auto
            self.dstrides = [2]*n_lay_auto
            self.gpaddings = [2]*(n_lay_auto)
            self.gpaddings[-1]= 3
            self.dpaddings = [1]*(n_lay_auto)
            self.dpaddings[-1] = 0
            self.gfilters,self.dfilters = [],[]
            f=512
            for i in range(len(self.gkernals)):
                self.gfilters.append(int(f))
                self.dfilters.insert(0,int(f))
                f/=2
            self.dfilters[-1] = 1
            self.dfilters.insert(0,self.channels)
            self.gfilters.append(self.channels)
            self.gfilters[0] = self.nz
            self.pred_out = self.training_image_size


    def train(self):
        ## Data Processing
        self.master.geometry('900x600')
        self.data_path = []


        if self.pro:
            self.isotropic = False
            for im in self.imgs: self.data_path.append(im.filename) # path to training data.


        else:
            self.isotropic = True
            self.data_path = [self.imgs[0].filename]

        self.data_type = os.path.splitext(self.data_path[0])[1][1:]  # png, jpg, tif, array, array2D
        self.Project_path =  os.path.splitext(self.data_path[0])[0] + '_results/'
        try:
            os.mkdir(self.Project_path)
        except:
            print('Overwriting...')
        ## Network Architectures

        ##Create Networks
        if self.pro:
            self.update_nets()
            self.netD, self.netG = Architect(self.Project_path, True, self.nets[1].kernals, self.nets[1].strides, self.nets[1].filters, self.nets[1].paddings,
                                   self.nets[0].kernals, self.nets[0].strides, self.nets[0].filters, self.nets[0].paddings)
        else:
            self.netD, self.netG = Architect(self.Project_path, True, self.dkernals, self.dstrides, self.dfilters,self.dpaddings,self.gkernals,self.gstrides,self.gfilters,self.gpaddings)
        Train.trainer(self.Project_path, self.image_type, self.data_type, self.data_path, self.netD, self.netG,
                      self.isotropic, self.channels, self.training_image_size, 32)





    def Training_graphs(self):
        try:
            for name, frame in zip(['_LossGraph.png','_WassGraph.png','_GpGraph.png','_slices.png'],self.trainingframes):
                im =Image.open(self.Project_path + name).convert('RGBA')
                render = ImageTk.PhotoImage(im)
                self.Tglbl = Label(frame, image=render)
                self.Tglbl.image = render
                self.Tglbl.grid(column=0, row=0)
        except:
            for frame in self.trainingframes:
                self.Tglbl = Label(frame, text = 'Please wait for first iteration to complete...')
                self.Tglbl.grid(column=0, row=0)





    def refresh(self):
        self.Training_graphs()
        self.master.update()
        self.master.after(1000, self.refresh)

    def pretrain_checks(self):
        if self.channels==2:
            self.image_type = 'twophase'  # threephase, twophase or colour
        elif self.channels ==3:
            self.image_type = 'threephase'
        else:
            return 'incorrect segmentation: provide a two or three phase image'
        if self.pro and self.pred_out != self.training_image_size:
            return 'Gen output: ' + str(self.pred_out) + ' and training image size: '+ str(self.training_image_size) +' do not match'
        return 'Training'

    def stop(self):
        self.master.destroy()

    def start(self):

        status = self.pretrain_checks()
        self.statuslab.config(text = 'status: ' + status)

        if status!= 'Training':
            return
        for child in self.f1.winfo_children():
            child.configure(state='disable')
        self.finishbut = Button(self.f1, text='Stop ', command=self.stop)
        self.finishbut.grid(column=7, row=7, sticky='sw')
        self.trainingframes = []
        for nme in ['Loss Graphs', 'Wasserstein Graph', 'Gradient Penalty', 'Example slices']:
            fr = Frame(self.nb)
            self.nb.add(fr, text=nme)
            self.trainingframes.append(fr)
        self.refresh()
        threading.Thread(target=self.train).start()

    class Img:
        def __init__(self, column, colour, outer_inst):
            self.loaded = False
            self.outer = outer_inst
            self.rotation = 0
            self.crp = None
            self.crplbl = 0
            self.scalefac = 1
            self.frame = outer_inst.f1
            self.column = column
            self.colour = colour
            self.browse = Button(self.frame, text='Browse Image ' + str(column + 1), command=lambda fn = None: self.get_filename(fn))
            self.browse.grid(column=column * 2+1, row=0, padx=(25, 0))
            self.dlt = Button(self.frame, text='X', fg='red', command=lambda cfn=True: self.del_image(cfn))
            self.dlt.grid(column=column * 2+1, row=0, padx=(150, 0))
            if self.outer.pro:
                self.rotlft = Button(self.frame, text='↻', command=lambda dir='ac': self.rot_image(dir))
                self.rotlft.grid(column=column * 2+1, row=2, padx=(0, 25))
                self.rotrt = Button(self.frame, text='↺', command=lambda dir='c': self.rot_image(dir))
                self.rotrt.grid(column=column * 2+1, row=2, padx=(80, 0))
            self.scaleEntry = Entry(self.frame, width=5)
            self.scaleEntry.grid(column=column * 2+1, row=1, padx=(150, 0))
            self.up_sf = Button(self.frame, text='set scale factor', command=self.set_scale_factor)
            self.up_sf.grid(column=column * 2+1, row=1, padx=(25, 0))

        def get_filename(self, fn):
            #open file browser or load file name if turning isotropic
            if not fn:
                self.filename = filedialog.askopenfilename()
            else:
                self.filename = fn
            #Save raw image
            self.img = Image.open(self.filename).convert('RGBA')
            #Delete previous entry if overriding
            if self.loaded:
                self.del_image(False)
            #Update to new image
            self.loaded = True
            self.update()

        def update(self):
            if self.loaded:

                if self.crplbl != 0:
                    self.del_image(False)
                im = self.img.rotate(self.rotation, expand=True)
                im = im.resize(size=(int(im.size[0] / self.scalefac), int(im.size[1] / self.scalefac)))
                crp = im.crop((0, 0, self.outer.training_image_size, self.outer.training_image_size))
                self.crp = ImageOps.expand(crp, border = int(self.outer.training_image_size/16), fill = self.colour)
                sf = max(im.size)*1.1 / self.outer.column_width
                im = ImageOps.expand(im.resize(size=(int(im.size[0] / sf), int(im.size[1] / sf))), border = 5, fill = self.colour)
                render = ImageTk.PhotoImage(im)
                self.imglbl = Label(self.frame, image=render)
                self.imglbl.image = render
                self.imglbl.grid(column=(self.column * 2+1), row=4, columnspan=1)
                sf = max(self.crp.size)*1.1 / self.outer.column_width
                Tkcrp = ImageOps.expand(crp.resize(size=(int(crp.size[0] / sf), int(crp.size[1] / sf))), border = 5, fill = self.colour)
                Tkcrp = ImageTk.PhotoImage(Tkcrp)
                self.crplbl = Label(self.frame, image=Tkcrp)
                self.crplbl.image = Tkcrp
                self.crplbl.grid(column=(self.column * 2+1), row=5, columnspan=1)
                self.outer.channels = np.unique(self.crp).size

            if not self.outer.pro:
                return
            for img in self.outer.imgs:
                if (not img.loaded) or (img.crp.size[0] != self.outer.training_image_size + int(self.outer.training_image_size/16)*2):
                    return
            self.cubeView()

        def del_image(self, cfn):
            # cfn: whether to clear filename
            if self.loaded:
                self.imglbl.config(image='')
                self.crplbl.config(image='')
            if cfn:
                self.filename = None
                self.loaded = False
            if self.outer.cube_loaded:
                self.outer.cubelabel.config(image = '')
            # set image labels to blank


        def set_scale_factor(self):
            if self.loaded:
                self.del_image(False)
                try:
                    self.scalefac = float(self.scaleEntry.get())
                except:
                    print('enter number')
                self.update()


        def rot_image(self, dir):
            if not self.loaded:
                return
            self.del_image(False)
            if dir == 'c':
                self.rotation += 90
            else:
                self.rotation += 270
            self.rot = self.rotation % 360
            self.update()

        def skew_image(self, img, angle, xsq, ysq):
            width, height = img.size
            xshift = np.tan(abs(angle)) * height
            new_width = width + int(xshift)
            if new_width < 0:
                return img
            img = img.transform((new_width, height), Image.AFFINE,
                                (1, angle, -xshift if angle > 0 else 0, 0, 1, 0), Image.BICUBIC)
            img = img.resize(size=(int(img.size[0] / xsq), int(img.size[1] / ysq)))
            return img

        def cubeView(self):
            im2 = self.skew_image(self.outer.imgs[1].crp, -0.45, 1, 1.7)
            im3 = self.skew_image(self.outer.imgs[2].crp, 0.6, 1, 2.08)
            im3 = im3.rotate(90, expand=1)
            l = self.crp.size[0]
            cube = np.zeros((im2.size[1] + l , im2.size[0], 4))
            cube[-l:, -l:, :] = self.outer.imgs[0].crp.crop((2,2,l-2,l-2)).resize(size = (l,l))
            cube[:im2.size[1], -im2.size[0]:] = im2
            im3arr = np.array(im3)
            cube[-im3.size[1]:, :im3.size[0]] += im3arr[:cube.shape[0]]
            sf = max(cube.shape) / self.outer.column_width
            cube = Image.fromarray(np.uint8(cube))
            cube = cube.resize(size=(int(cube.size[0] / sf), int(cube.size[1] / sf)))
            if self.outer.cube_loaded:
                self.outer.cubelabel.config(image='')
            self.outer.cube_loaded = True
            self.outer.cube = ImageTk.PhotoImage(cube)
            self.outer.cubelabel = Label(self.frame, image=self.outer.cube)
            self.outer.cubelabel.image = self.outer.cube
            self.outer.cubelabel.grid(column=(0), row=4, rowspan=2)

    class NetImg:
        def __init__(self,row0, type, outer_inst):
            self.loaded = False
            self.outer = outer_inst
            self.row0 = row0
            self.type = type
            self.set_params()
            self.f2 = self.outer.f2
            self.addlay = Button(self.f2, text='Add Layer', command=self.add_layer)
            self.addlay.grid(column=0, row=row0+5)
            self.addlay = Button(self.f2, text='Remove Layer', command=self.rem_layer)
            self.addlay.grid(column=1, row=row0+5 )
            for txt, row, col in zip(['strides:','kernals:', 'paddings:', 'filters:'], [1, 1, 1, 1],
                                     [ 0+row0, 1+row0, 2+row0, 3+row0]):
                lab = Label(self.f2, text=txt)
                lab.grid(row=col, column=row)
            lab = Label(self.f2, text=type + ':', font = font.Font(family="Lucida Grande", size=12))
            lab.grid(row=row0-1, column=0, columnspan = 3, pady = (30,0),sticky = 'w')
            self.ks = []
            self.ss = []
            self.ps = []
            self.fs = []
            self.netG_lbls = []
            self.inp = 4
            self.lay_num = len(self.kernals)
            for col in range(self.lay_num):
                self.add_button(col)

        def set_params(self):
            if self.type[0] == 'G':
                self.kernals = self.outer.gkernals[:]
                self.paddings = self.outer.gpaddings[:]
                self.strides = self.outer.gstrides[:]
                self.filters = self.outer.gfilters[:]
            else:
                self.kernals = self.outer.dkernals[:]
                self.paddings = self.outer.dpaddings[:]
                self.strides = self.outer.dstrides[:]
                self.filters = self.outer.dfilters[:]
        def convolve(self, k, s, p):
            if self.type[0] == 'G':
                inp = torch.ones(1, 1, self.inp, self.inp)
                conv = nn.ConvTranspose2d(1, 1, k, s, p, bias=False)
            else:
                inp = torch.ones(1, 1, self.outer.training_image_size, self.outer.training_image_size)
                conv = nn.Conv2d(1, 1, k, s, p, bias=False)
            conv.weight[0][0] = 1
            return conv(inp)

        def netG(self, ks, ss, ps):
            outputs = []
            if self.type[0] == 'G':
                input = torch.ones(1, 1, self.inp, self.inp)
            else:
                input = torch.ones(1, 1, self.outer.training_image_size, self.outer.training_image_size)
            outputs.append(input.detach().numpy().astype('int')[0, 0])

            for k, s, p in zip(ks, ss, ps):
                if self.type[0] == 'G':
                    conv = nn.ConvTranspose2d(1, 1, k, s, p, bias=False)
                else:
                    conv = nn.Conv2d(1, 1, k, s, p, bias=False)
                conv.weight[0][0] = 1
                input = conv(input)
                outputs.append(input.detach().numpy().astype('int')[0, 0])
            return outputs

        def add_button(self, col):
            for r, (button, param) in enumerate(zip([self.ss, self.ks, self.ps, self.fs],
                                                    [self.strides, self.kernals, self.paddings, self.filters])):
                if r!= 3:
                    box = Entry(self.f2, width=2)
                    box.grid(column=col*2 + 3, row=r+self.row0, sticky = 'w')
                else:
                    box = Entry(self.f2, width=3)
                    box.grid(column=col*2 + 2, row=r + self.row0)
                box.insert(0, param[col])
                button.append(box)

        def update_arch(self, event=None):
            for lbl in self.netG_lbls:
                lbl.destroy()
            self.kernals = []
            self.strides = []
            self.paddings = []
            self.filters = []
            for s, k, p, f in zip(self.ss, self.ks, self.ps, self.fs):
                self.strides.append(int(s.get()))
                self.kernals.append(int(k.get()))
                self.paddings.append(int(p.get()))
                self.filters.append(int(f.get()))
            if self.type[0] == 'G':
                self.filters.append(self.outer.channels)
            else:
                self.filters.append(1)
            self.show_arch()

        def show_arch(self):
            layers = self.netG(self.kernals, self.strides, self.paddings)
            self.netG_lbls = []
            for i, lay in enumerate(layers):
                title = 'input: ' if i ==0 else 'out size: '

                im = Image.fromarray(lay * 255 / np.max(lay))
                width = int(np.min((400 / self.outer.net_imwidth, 100)))
                im = im.resize(size=(width, width))
                im = ImageTk.PhotoImage(im)
                lbl = Label(self.f2, text = title + str(lay.shape[0]), image=im, compound='bottom')
                lbl.image = im
                lbl.grid(column=i*2 + 2, row=self.row0+5)
                self.netG_lbls.append(lbl)
            if self.type[0] == 'G':
                self.outer.pred_out = lay.shape[0]

        def add_layer(self):
            self.kernals.append(4)
            self.strides.append(2)
            self.paddings.append(0)
            self.filters.append(0)
            self.add_button(self.lay_num)
            self.lay_num += 1
            self.outer.update_nets()

        def rem_layer(self):
            if self.lay_num < 3:
                return
            for button, param in zip([self.ss, self.ks, self.ps, self.fs],
                                     [self.strides, self.kernals, self.paddings, self.filters]):
                param.pop()
                button[-1].destroy()
                button.pop()
            self.lay_num -= 1

            self.outer.update_nets()


root = Tk()
app = Window(root)
app.mainloop()
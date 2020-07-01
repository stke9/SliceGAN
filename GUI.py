from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import numpy as np
from copy import deepcopy as dc
import matplotlib as plt

class Window(Frame):
    def __init__(self, master = None):
        Frame.__init__(self, master)
        self.master = master
        self.init_window()
        self.master.geometry('800x600')

    def init_window(self):
        self.master.title('SliceGAN')
        self.nb = ttk.Notebook(self.master)
        self.nb.pack(side = TOP, fill = BOTH, expand = 1)
        self.master.bind('<Configure>', self.resize_with_window)
        ##First frame: Training image Loading
        self.f1 = Frame(self.nb)
        self.column_width = self.f1.winfo_width()/4
        self.imgs = []
        self.training_image_size = 64
        self.cube_loaded = False
        #iso button for isotropy
        self.iso = IntVar()
        self.isobutton =  Checkbutton(self.f1, text='Isotropic', onvalue=1, offvalue=0, command = self.turn_Iso, variable = self.iso)
        self.isobutton.grid(column = 7, row = 0)
        #Entry o update all image widths
        self.sizeEntry = Entry(self.f1, width=5)
        self.sizeEntry.grid(column=7, row=1, padx=(150, 0))
        self.up_size = Button(self.f1, text='set image size', command=self.set_img_sizes)
        self.up_size.grid(column=7, row=1, padx=(25, 0))
        #Close window and save
        self.finishbut = Button(self.f1, text = 'Finish',  command = self.finish)
        self.finishbut.grid(column = 7, row = 7, sticky = 'se')
        #Row configs for resizing
        for c in range(7):
            self.f1.grid_columnconfigure(c, weight=1)
        self.f1.grid_rowconfigure(7, weight=1)
        #Initiate Image instances
        for column, colour in zip(range(3), ['red','green','blue']):
            self.imgs.append(self.Img(column,colour, self))

        ## Init page 2
        self.f2 = Frame(self.nb)
        self.button = Button(self.f2, text = 'Browse')
        self.button.grid(column = 0, row = 0)

        ## add pages to notbook
        self.nb.add(self.f1, text = 'Image Load')
        self.nb.add(self.f2, text = 'Architecture')
        self.nb.select(self.f1)

    def resize_with_window(self, event):
        if (abs(self.column_width - self.f1.winfo_width()/4)>20):
            self.column_width = self.f1.winfo_width()/4
            self.update_images()

    def update_images(self):
        for img in self.imgs:
            img.update()

    def finish(self):
        self.master.destroy()

    def turn_Iso(self):
        if self.iso and self.imgs[0].loaded:
            for i,img in enumerate(self.imgs[1:]):
                img.get_filename(self.imgs[0].filename)
        self.update_images()

    def set_img_sizes(self):
        try :
            self.training_image_size = float(self.sizeEntry.get())
            self.update_images()
        except:
            print('enter number')
            return



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
            self.browse.grid(column=column * 2, row=0, padx=(25, 0))
            self.dlt = Button(self.frame, text='X', fg='red', command=lambda cfn=True: self.del_image(cfn))
            self.dlt.grid(column=column * 2, row=0, padx=(150, 0))
            self.rotlft = Button(self.frame, text='↻', command=lambda dir='ac': self.rot_image(dir))
            self.rotlft.grid(column=column * 2, row=1, padx=(0, 25))
            self.rotrt = Button(self.frame, text='↺', command=lambda dir='c': self.rot_image(dir))
            self.rotrt.grid(column=column * 2, row=1, padx=(80, 0))
            self.scaleEntry = Entry(self.frame, width=5)
            self.scaleEntry.grid(column=column * 2, row=2, padx=(150, 0))
            self.up_sf = Button(self.frame, text='set scale factor', command=self.set_scale_factor)
            self.up_sf.grid(column=column * 2, row=2, padx=(25, 0))

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
                self.imglbl.grid(column=(self.column * 2), row=4, columnspan=1)
                sf = max(self.crp.size)*1.1 / self.outer.column_width
                Tkcrp = ImageOps.expand(crp.resize(size=(int(crp.size[0] / sf), int(crp.size[1] / sf))), border = 5, fill = self.colour)
                Tkcrp = ImageTk.PhotoImage(Tkcrp)
                self.crplbl = Label(self.frame, image=Tkcrp)
                self.crplbl.image = Tkcrp
                self.crplbl.grid(column=(self.column * 2), row=5, columnspan=1)

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
            self.outer.cubelabel.grid(column=(7), row=4, rowspan=2)

root = Tk()
app = Window(root)
app.mainloop()
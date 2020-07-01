from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

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
        self.clicked = False
        # self.master.bind('<ButtonRelease-1>', self.resize)
        self.master.bind('<Configure>', self.resize)
        ## Init page 1
        self.f1 = Frame(self.nb)
        self.imgs = []
        self.isobutton =  Checkbutton(self.f1, text='Isotropic', onvalue=1, offvalue=0, command = self.turn_Iso)
        self.isobutton.grid(column = 7, row = 0)
        self.sizeEntry = Entry(self.f1, width=5)
        self.sizeEntry.grid(column=7, row=1, padx=(150, 0))
        self.up_size = Button(self.f1, text='set image size', command=self.set_img_sizes)
        self.up_size.grid(column=7, row=1, padx=(25, 0))
        self.finishbut = Button(self.f1, text = 'Finish',  command = self.finish)
        self.finishbut.grid(column = 7, row = 7, sticky = 'se')
        for c in range(7):
            self.f1.grid_columnconfigure(c, weight=1)
        self.f1.grid_rowconfigure(7, weight=1)

        for col in range(3):
            self.imgs.append(img(col, self.f1))

        ## Init page 2
        self.f2 = Frame(self.nb)
        self.button = Button(self.f2, text = 'Browse')
        self.button.grid(column = 0, row = 0)

        ## add pages to notbook
        self.nb.add(self.f1, text = 'Image Load')
        self.nb.add(self.f2, text = 'Architecture')
        self.nb.select(self.f1)

        # self.nb.enable_traversal()
    def resize(self, event):
        for im in self.imgs:
            if (im.loaded) and (abs(im.width - im.frame.winfo_width()/4)>5):
                im.del_image(False)
                im.update_image()

    def finish(self):
        self.master.destroy()

    def turn_Iso(self):

        for img in self.imgs[1:]:
             if (img.browse['state'] == NORMAL):
                 for but in [img.browse, img.rotlft, img.rotrt, img.scaleEntry, img.up_sf]:
                    but['state'] = DISABLED
                 disabled = True
             else:
                 for but in [img.browse, img.rotlft, img.rotrt, img.scaleEntry, img.up_sf]:
                     but['state'] = NORMAL
                 disabled = False

             if disabled:
                 img.del_image(False)
             else:
                 if img.loaded: img.update_image()

    def set_img_sizes(self):
        try :
            newsize = float(self.sizeEntry.get())
        except:
            print('enter number')
            return

        for im in self.imgs:
            im.set_imsize(newsize)



class img():
    def __init__(self, col, frm):
        self.loaded = False
        self.rotation = 0
        self.scalefac = 1
        self.frame = frm
        self.col = col
        self.imsize = 64
        self.browse = Button(frm, text = 'Browse Image ' + str(col+1), command = self.get_img)
        self.browse.grid(column = col*2, row = 0, padx = (25,0))
        self.dlt = Button(frm, text='X', fg = 'red', command = lambda cfn = True: self.del_image(cfn))
        self.dlt.grid(column=col*2, row=0, padx=(150,0))
        self.rotlft = Button(frm, text='↻', command=lambda dir='ac': self.rot_image(dir))
        self.rotlft.grid(column=col*2, row=1, padx=(0, 25))
        self.rotrt = Button(frm, text='↺', command=lambda dir='c': self.rot_image(dir))
        self.rotrt.grid(column=col*2, row=1, padx=(80, 0))
        self.scaleEntry = Entry(self.frame, width = 5)
        self.scaleEntry.grid(column=col*2, row=2, padx=(150,0))
        self.up_sf= Button(self.frame, text='set scale factor', command = self.set_SF)
        self.up_sf.grid(column=col*2, row=2, padx=(25, 0))

    def set_SF(self):
        if self.loaded:
            self.del_image(False)
            try:
                self.scalefac = float(self.scaleEntry.get())
            except:
                print('enter number')
            self.update_image()
    def set_imsize(self, size):
        self.imsize = size
        if self.loaded:
            self.del_image(False)
            self.update_image()

    def update_image(self):
        im = self.img.rotate(self.rotation, expand = True)
        im = im.resize(size = (int(im.size[0]/self.scalefac), int(im.size[1]/self.scalefac)))
        crp = im.crop((0, 0, self.imsize, self.imsize))

        self.width = self.frame.winfo_width()/4
        sf = max(im.size)/self.width
        im = im.resize(size = (int(im.size[0]/sf), int(im.size[1]/sf)))
        render = ImageTk.PhotoImage(im)
        self.imglbl = Label(self.frame, image = render)
        self.imglbl.image = render
        self.imglbl.grid(column = (self.col*2), row = 4, columnspan = 1)

        sf = max(crp.size) / self.width
        crp = crp.resize(size = (int(crp.size[0]/sf), int(crp.size[1]/sf)))
        render = ImageTk.PhotoImage(crp)
        self.crplbl = Label(self.frame, image=render)
        self.crplbl.image = render
        self.crplbl.grid(column=(self.col * 2), row=5, columnspan=1)

    def get_img(self):
        self.filename = filedialog.askopenfilename()
        self.img = Image.open(self.filename)
        if self.loaded:
            self.del_image(False)
        self.update_image()
        self.loaded = True

    def rot_image(self,dir):
        if self.loaded != True:
            return
        if self.filename == None:
            return
        self.del_image(False)
        if dir =='c':
            self.rotation +=90
        else:
            self.rotation +=270
        self.rot = self.rotation%360
        self.update_image()


    def del_image(self,cfn):
        if cfn:
            self.filename = None
        if self.loaded:
            self.imglbl.config(image = '')
            self.crplbl.config(image='')

root = Tk()
app = Window(root)
app.mainloop()
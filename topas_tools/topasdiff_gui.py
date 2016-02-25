from Tkinter import *
from tkFileDialog import *
from ttk import *

import os

from collections import namedtuple

import topasdiff

def which(program):
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


class topasdiffDialog(Tk, object):

    """Dialog that provide settings window for Topasdiff"""

    def __init__(self, parent, drc='.'):
        super(topasdiffDialog, self).__init__()

        self.parent = parent

        self.init_vars()

        self.drc = '.'

        self.title("GUI for topasdiff")

        body = Frame(self, padding=(10, 10, 10, 10))
        self.initial_focus = self.body(body)
        body.columnconfigure(0, weight=1)
        body.pack(fill=BOTH, anchor=CENTER, expand=True)

        self.buttonbox()
        
        self.grab_set()

        self.update()
        self.geometry(self.geometry())       

    def init_vars(self):
        self.cif_file = StringVar()
        self.cif_file.set("structure.cif")

        self.fobs_file = StringVar()
        self.fobs_file.set("fobs.hkl")

        self.scale = DoubleVar()
        self.scale.set(1.0)

        self.superflip_path = StringVar()

        self.run_superflip = BooleanVar()
        self.run_superflip.set(True)

    def body(self, master):
        
        lfcif    = Labelframe(master, text="Path to cif file", padding=(10, 10, 10, 10))
        self.e_fname = Entry(
            lfcif, textvariable=self.cif_file)
        self.e_fname.grid(row=11, column=0, columnspan=3, sticky=E+W)
        but_load = Button(lfcif, text="Browse..", width=10, command=self.load_cif_file)
        but_load.grid(row=11, column=4, sticky=E)
        lfcif.grid(row=0, sticky=E+W)
        lfcif.columnconfigure(0, minsize=120)
        lfcif.columnconfigure(0, weight=1)

        lffobs   = Labelframe(master, text="Path to observed structure factors file", padding=(10, 10, 10, 10))
        self.e_fname = Entry(
            lffobs, textvariable=self.fobs_file)
        self.e_fname.grid(row=21, column=0, columnspan=3, sticky=E+W)
        but_load = Button(lffobs, text="Browse..", width=10, command=self.load_fobs_file)
        but_load.grid(row=21, column=4, sticky=E)
        lffobs.grid(row=1, sticky=E+W)
        lffobs.columnconfigure(0, minsize=120)
        lffobs.columnconfigure(0, weight=1)

        Label(lffobs, text="Scale").grid(row=25, column=0, sticky=W)
        self.e_fname = Entry(
            lffobs, textvariable=self.scale)
        self.e_fname.grid(row=25, column=1, sticky=W)

        if which("superflip"):
            self.superflip_path.set("superflip")
        elif which("superflip.exe"):
            self.superflip_path.set("superflip.exe")
        else:
            lfsf   = Labelframe(master, text="Path to superflip executable", padding=(10, 10, 10, 10))
            self.e_fname = Entry(
                lfsf, textvariable=self.superflip_path)
            self.e_fname.grid(row=31, column=0, columnspan=3, sticky=E+W)
            but_load = Button(lfsf, text="Browse..", width=10, command=self.set_superflip_path)
            but_load.grid(row=31, column=4, sticky=E)
            lfsf.grid(row=2, sticky=E+W)
            lfsf.columnconfigure(0, minsize=120)
            lfsf.columnconfigure(0, weight=1)

        # self.c_run_superflip = Checkbutton(lfsf, variable=self.run_superflip, text="Run superflip?")
        # self.c_run_superflip.grid(row=32, column=0, sticky=W)


    def buttonbox(self):
        box = Frame(self)

        w = Button(box, text="Run", width=10, command=self.ok, default=ACTIVE)
        w.pack(side=RIGHT, padx=5, pady=5)
        w = Button(box, text="Exit", width=10, command=self.cancel)
        w.pack(side=RIGHT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

        box.pack(fill=X, anchor=S, expand=False)

    def add_cell(self):
        var = self.init_cell()
        self.vars.append(var)
        self.page(var)

    def ok(self, event=None):
        if not self.validate():
            return

        # self.withdraw()
        self.update_idletasks()

        self.apply()

        # self.destroy()

    def cancel(self, event=None):
        self.destroy()

    def validate(self):
        return 1  # override

    def apply(self):
        gui_options = {
            "args": self.cif_file.get(),
            "diff": self.fobs_file.get(),
            "superflip_path": self.superflip_path.get(),
            "run_superflip": self.run_superflip.get(),
            "scale": self.scale.get()
        }
        topasdiff.run_script(gui_options=gui_options)

    def load_cif_file(self):

        f = askopenfilename(initialdir=self.drc)
        if f:
            self.cif_file.set(str(f))

    def load_fobs_file(self):
        f = askopenfilename(initialdir=self.drc)
        if f:
            self.fobs_file.set(str(f))

    def set_superflip_path(self):
        f = askopenfilename(initialdir=self.drc)
        if f:
            self.superflip_path.set(str(f))

def run():
    app = topasdiffDialog(None)
    app.mainloop()

if __name__ == '__main__':
    print run()

from Tkinter import *
from tkFileDialog import *
from ttk import *

import subprocess as sp
import os

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

    """Dialog that provide settings window for crystal parameters"""

    def __init__(self, parent, drc='.'):
        super(topasdiffDialog, self).__init__()

        self.parent = parent

        self.init_vars()

        self.drc = '.'

        self.title("GUI for topasdiff")

        body = Frame(self, padding=(10, 10, 10, 10))
        self.initial_focus = self.body(body)
        body.columnconfigure(1, weight=1)
        body.columnconfigure(3, weight=1)
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
        Label(master, text="Path to cif file").grid(row=10, sticky=W)
        self.e_fname = Entry(
            master, textvariable=self.cif_file)
        self.e_fname.grid(row=11, column=0, columnspan=3, sticky=E+W)

        Label(master, text="Path to observed structure factors").grid(row=20, sticky=W)
        self.e_fname = Entry(
            master, textvariable=self.fobs_file)
        self.e_fname.grid(row=21, column=0, columnspan=3, sticky=E+W)

        Label(master, text="Scale").grid(row=25, sticky=W)
        self.e_fname = Entry(
            master, textvariable=self.scale)
        self.e_fname.grid(row=26, column=0, columnspan=3, sticky=E+W)

        if not which("superflip"):
            Label(master, text="Path to superflip").grid(row=30, sticky=W)
            self.e_fname = Entry(
                master, textvariable=self.superflip_path)
            self.e_fname.grid(row=31, column=0, columnspan=3, sticky=E+W)

        self.c_run_superflip = Checkbutton(master, variable=self.run_superflip, text="Run superflip?")
        self.c_run_superflip.grid(row=32, column=0, sticky=W)

        but_load = Button(master, text="Browse..", width=10, command=self.load_cif_file)
        but_load.grid(row=11, column=4, sticky=E)
        but_load = Button(master, text="Browse..", width=10, command=self.load_fobs_file)
        but_load.grid(row=21, column=4, sticky=E)
        but_load = Button(master, text="Browse..", width=10, command=self.set_superflip_path)
        but_load.grid(row=31, column=4, sticky=E)

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
        print "cif_file", self.cif_file.get()
        print "fobs_file", self.fobs_file.get()
        print "scale", self.scale.get()

        print "superflip_path", self.superflip_path.get(), which(self.superflip_path.get())
        print "run_superflip", self.run_superflip.get()


    def load_cif_file(self):
        f = askopenfile(initialdir=self.drc)
        if f:
            self.cif_file.set(f)

    def load_fobs_file(self):
        f = askopenfile(initialdir=self.drc)
        if f:
            self.fobs_file.set(f)

    def set_superflip_path(self):
        f = askopenfile(initialdir=self.drc)
        if f:
            self.superflip_path.set(f)

def main():
    app = topasdiffDialog(None)
    app.mainloop()

if __name__ == '__main__':
    main()

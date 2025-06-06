import os
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename

from future import standard_library

from . import topasdiff

standard_library.install_aliases()


def which(program):
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ['PATH'].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


class topasdiffDialog(tk.Tk):
    """Dialog that provide settings window for Topasdiff"""

    def __init__(self, parent, drc='.'):
        super().__init__()

        self.parent = parent

        self.init_vars()

        self.drc = '.'

        self.title('GUI for topasdiff')

        body = ttk.Frame(self, padding=(10, 10, 10, 10))
        self.initial_focus = self.body(body)
        body.columnconfigure(0, weight=1)
        body.pack(fill=tk.BOTH, anchor=tk.CENTER, expand=True)

        self.buttonbox()

        self.grab_set()

        self.update()
        self.geometry(self.geometry())

    def init_vars(self):
        self.cif_file = tk.StringVar()
        self.cif_file.set('structure.cif')

        self.fobs_file = tk.StringVar()
        self.fobs_file.set('fobs.hkl')

        self.scale = tk.DoubleVar()
        self.scale.set(1.0)

        self.superflip_path = tk.StringVar()

        self.run_superflip = tk.BooleanVar()
        self.run_superflip.set(True)

    def body(self, master):
        lfcif = ttk.Labelframe(master, text='Path to cif file', padding=(10, 10, 10, 10))
        self.e_fname = ttk.Entry(lfcif, textvariable=self.cif_file)
        self.e_fname.grid(row=11, column=0, columnspan=3, sticky=tk.E + tk.W)
        but_load = ttk.Button(lfcif, text='Browse..', width=10, command=self.load_cif_file)
        but_load.grid(row=11, column=4, sticky=tk.E)
        lfcif.grid(row=0, sticky=tk.E + tk.W)
        lfcif.columnconfigure(0, minsize=120)
        lfcif.columnconfigure(0, weight=1)

        lffobs = ttk.Labelframe(
            master, text='Path to observed structure factors file', padding=(10, 10, 10, 10)
        )
        self.e_fname = ttk.Entry(lffobs, textvariable=self.fobs_file)
        self.e_fname.grid(row=21, column=0, columnspan=3, sticky=tk.E + tk.W)
        but_load = ttk.Button(lffobs, text='Browse..', width=10, command=self.load_fobs_file)
        but_load.grid(row=21, column=4, sticky=tk.E)
        lffobs.grid(row=1, sticky=tk.E + tk.W)
        lffobs.columnconfigure(0, minsize=120)
        lffobs.columnconfigure(0, weight=1)

        tk.Label(lffobs, text='Scale').grid(row=25, column=0, sticky=tk.W)
        self.e_fname = ttk.Entry(lffobs, textvariable=self.scale)
        self.e_fname.grid(row=25, column=1, sticky=tk.W)

        if which('superflip'):
            self.superflip_path.set('superflip')
        elif which('superflip.exe'):
            self.superflip_path.set('superflip.exe')
        else:
            lfsf = ttk.Labelframe(
                master, text='Path to superflip executable', padding=(10, 10, 10, 10)
            )
            self.e_fname = ttk.Entry(lfsf, textvariable=self.superflip_path)
            self.e_fname.grid(row=31, column=0, columnspan=3, sticky=tk.E + tk.W)
            but_load = ttk.Button(
                lfsf, text='Browse..', width=10, command=self.set_superflip_path
            )
            but_load.grid(row=31, column=4, sticky=tk.E)
            lfsf.grid(row=2, sticky=tk.E + tk.W)
            lfsf.columnconfigure(0, minsize=120)
            lfsf.columnconfigure(0, weight=1)

    def buttonbox(self):
        box = ttk.Frame(self)

        w = ttk.Button(box, text='Run', width=10, command=self.ok, default=tk.ACTIVE)
        w.pack(side=tk.RIGHT, padx=5, pady=5)
        w = ttk.Button(box, text='Exit', width=10, command=self.cancel)
        w.pack(side=tk.RIGHT, padx=5, pady=5)

        self.bind('<Return>', self.ok)
        self.bind('<Escape>', self.cancel)

        box.pack(fill=tk.X, anchor=tk.S, expand=False)

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
        os.chdir(self.drc)
        gui_options = {
            'args': self.cif_file.get(),
            'diff': self.fobs_file.get(),
            'superflip_path': self.superflip_path.get(),
            'run_superflip': self.run_superflip.get(),
            'scale': self.scale.get(),
        }
        topasdiff.run_script(gui_options=gui_options)

    def load_cif_file(self):
        f = askopenfilename(initialdir=self.drc)
        if f:
            self.cif_file.set(str(f))
            self.drc = os.path.dirname(f)

    def load_fobs_file(self):
        f = askopenfilename(initialdir=self.drc)
        if f:
            self.fobs_file.set(str(f))
            self.drc = os.path.dirname(f)

    def set_superflip_path(self):
        f = askopenfilename(initialdir=self.drc)
        if f:
            self.superflip_path.set(str(f))


def run():
    app = topasdiffDialog(None)
    app.mainloop()


if __name__ == '__main__':
    print(run())

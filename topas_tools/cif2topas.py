import sys
cifs = sys.argv[1:]

from .cif import reader, CifParserError


def read_cif(f, verbose=False):
    """opens cif and returns cctbx data object"""
    try:
        if isinstance(f, file):
            structures = reader(file_object=f).build_crystal_structures()
        elif isinstance(f, str):
            structures = reader(file_path=f).build_crystal_structures()
        else:
            raise TypeError(
                f'read_cif: Can not deal with type {type(f)}')
    except CifParserError as e:
        print(e)
        print("Error parsing cif file, check if the data tag does not contain any spaces.")
        sys.exit()
    if verbose:
        for key, val in list(structures.items()):
            print("\nstructure:", key)
            val.show_summary().show_scatterers()
    return structures


def main():
    if len(cifs) == 0:
        print("cif2topas - tool for converting cif files to Topas format (.inp)")
        print("Takes any number of cif files and writes to stdout")
        print("")
        print("usage: cif2topas cif1 [cif2 ...]")
        sys.exit()

    for cif in cifs:
        # print 'Reading file: {}'.format(cif)
        structures = read_cif(cif)
        for name in structures:
            structure = structures[name].sites_mod_positive()
            sg = structure.space_group()
            uc = structure.unit_cell()
            sps = structure.special_position_settings()
            scatterers = structure.scatterers()

            print("""
   str
      Out_X_Yobs(x_yobs.xy)
      Out_X_Ycalc(x_ycalc.xy)
      Out_X_Difference(x_ydiff.xy)
      Create_2Th_Ip_file(ticks.out)

      Out_fobs(fobs.hkl)

      Out_CIF_STR(structure.cif)""")

            print("""       
      scale  @  1.0
      r_bragg  1.0""")
            print()

            s400 = "      'prm s400 1.0 \n"
            s040 = "      'prm s040 1.0 \n"
            s004 = "      'prm s004 1.0 \n"
            s220 = "      'prm s220 1.0 \n"
            s202 = "      'prm s202 1.0 \n"
            s022 = "      'prm s022 1.0 \n"
            s301 = "      'prm s301 1.0 \n"
            s121 = "      'prm s121 1.0 \n"
            s103 = "      'prm s103 1.0 \n"
            eta  = "      'prm !eta 0.5 min 0.0 max 1.0\n"
            if sg.crystal_system() == 'Monoclinic':
                string = "{s400}{s040}{s004}{s220}{s202}{s022}{s301}{s121}{s103}{eta}".format(
                    s400=s400, s040=s040, s004=s004, s220=s220, s202=s202, s022=s022, s301=s301, s121=s121, s103=s103, eta=eta)
                macro = "      'Stephens_monoclinic(s400, s040, s004, s220, s202, s022, s301, s121, s103, eta)"

            elif sg.crystal_system() == 'Tetragonal':
                string = "{s400}{s004}{s220}{s202}{eta}".format(
                    s400=s400, s040=s040, s004=s004, s220=s220, s202=s202, s022=s022, s301=s301, s121=s121, s103=s103, eta=eta)
                macro = "      'Stephens_tetragonal(s400, s004, s220, s202, eta)"

            elif sg.crystal_system() == 'Hexagonal':
                string = "{s400}{s004}{s202}{eta}".format(
                    s400=s400, s040=s040, s004=s004, s220=s220, s202=s202, s022=s022, s301=s301, s121=s121, s103=s103, eta=eta)
                macro = "      'Stephens_hexagonal(s400, s202, s004, eta)"

            elif sg.crystal_system() == 'Orthorhombic':
                string = "{s400}{s040}{s004}{s220}{s202}{s022}{eta}".format(
                    s400=s400, s040=s040, s004=s004, s220=s220, s202=s202, s022=s022, s301=s301, s121=s121, s103=s103, eta=eta)
                macro = "      'Stephens_orthorhombic(s400, s040, s004, s220, s202, s022, eta)"
            else:
                string = ""
                macro = ""
            print(string, end=' ')
            print(macro)

            print("""
      PV_Peak_Type(
      ha,    0.02,
      !hb,   0.0,
      !hc,   0.0,
      lora,  0.5,
      !lorb, 0.0,
      !lorc, 0.0)

      'Simple_Axial_Model(axial, 0.0)

      Phase_Density_g_on_cm3( 1.0)""")

            print("""
      view_structure

      'fourier_map 1
      '   fourier_map_formula = Fobs - Fcalc;""")
            print()
            print('      space_group "{}"'.format(sg.type().universal_hermann_mauguin_symbol().replace(" ", "")))
            print()
            a, b, c, al, be, ga = uc.parameters()
            if sg.crystal_system() == 'Cubic':
                string = "      Cubic(@ {a})"

            elif sg.crystal_system() == 'Tetragonal':
                string = "      Tetragonal(@ {a}, @ {c})"

            elif sg.crystal_system() == 'Hexagonal':
                string = "      Hexagonal(@ {a}, @ {c})"

            elif sg.crystal_system() == 'Rhombohedral':
                string = "      Rhombohedral(@ {a}, @ {al})"
            else:
                string = """      a  @  {a:8.5f}
      b  @  {b:8.5f}
      c  @  {c:8.5f}
      al {refal:1}  {al}
      be {refbe:1}  {be}
      ga {refga:1}  {ga}"""
            refal = "" if al == 90 else "@"
            refbe = "" if be == 90 else "@"
            refga = "" if ga == 90 else "@"
            print(string.format(a=a, b=b, c=c, refal=refal, al=al, refbe=refbe, be=be, refga=refga, ga=ga))
            print()
            print(f"      volume {uc.volume():.2f}")
            print()
            for element in set(scatterers.extract_scattering_types()):
                print(f"      prm beq{element:2s}  2.0  min 1.0  max 5.0")
            print()

            z_order = sg.order_z()

            for atom in scatterers:
                label = atom.label
                x, y, z = atom.site
                atom.element_symbol()
                element = atom.element_symbol()
                mult = atom.multiplicity()

                print("      site {label:5s}  num_posns {mult:3d}  x  {x:.5f}  y  {y:.5f}  z  {z:.5f}  occ {element:2s}  1.0  beq =beq{element:2s};".format(label=label, mult=mult, x=x, y=y, z=z, element=element), end=' ')

                if mult < z_order:
                    print("   ' {:5s} {:5s} {:5s}".format(*sps.site_symmetry(atom.site).special_op_simplified().terms))
                else:
                    print()


if __name__ == '__main__':
    main()

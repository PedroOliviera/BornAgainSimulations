import bornagain as ba

def get_materials():
    material_PS = ba.RefractiveMaterial("PS", 1.50267703E-06, 2.46904652E-09)
    material_P2VP = ba.RefractiveMaterial("P2VP", 2.51436745E-06, 2.35391329E-09)
    material_Si_Sub = ba.RefractiveMaterial("Si Sub", 5.04383115E-06, 7.84182177E-08)  # 7.644e-06
    material_SiO2 = ba.RefractiveMaterial("SiO2", 4.74631315E-06, 4.16025294E-08)
    material_Vacuum = ba.RefractiveMaterial("Vacuum", 0.0, 0.0)

    return dict(
        PS=material_PS,
        P2VP=material_P2VP,
        Si_Sub=material_Si_Sub,
        SiO2=material_SiO2,
        Vacuum=material_Vacuum,
    )

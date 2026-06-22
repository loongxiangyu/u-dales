from types import SimpleNamespace
from udprep.udprep_radiation import RadiationSection
import unittest
import numpy as np
   
   
class Property:
    def __init__(self) -> None:
        self.glaz = SimpleNamespace(
            T_0=np.array([0.775, 0.775, 0.775]),
            Rf_0=np.array([0.071, 0.071, 0.071]),
            Rb_0=np.array([0.071, 0.071, 0.071]),
            d_g=np.array([0.006, 0.006, 0.006]),
            d_gas=np.array([0.01, 0.01]),
        )

class TestGlazing(unittest.TestCase): 
    def test_glazing_properties(self):
        # expected value
        Rw_dir=0.1608
        Rw_dif=0.2398
        knet_layer1=41.5823
        
        # three facets: two glazing facets (type 30) and one non-glazing facet (type 1)
        svf = np.array([1, 1, 1])
        phi = np.deg2rad(np.array([45, 45, 90]))
        facet_types = np.array([30, 1, 30])
        sdir = np.array([240.51, 240.51, 0.0])
        dsky = 246.73
        albedo = np.array([0.2, 0.2, 0.2])
        vf = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=float,
        )
        
        values = {}
        sim = Property()
        solver = RadiationSection("radiation", values, sim=sim)
        sdir=np.cos(phi)*sdir
        nglaz=np.sum(facet_types == 30)
        knet, knet_glaz, al_glaz, solar = solver.calc_knet_glaz(
            sdir, dsky, albedo, vf, svf, phi, facet_types
        )
        print("knet_glaz:", knet_glaz)
        print("al_glaz:", al_glaz)
        print("knet:", knet)
        # self.assertEqual(knet_glaz.shape[0], nglaz)
        # self.assertAlmostEqual(al_glaz[0][1], Rw_dir, delta=0.0001)
        # self.assertAlmostEqual(al_glaz[0][2], Rw_dif, delta=0.0001)
        # self.assertAlmostEqual(knet_glaz[0][1], knet_layer1, delta=0.0001)
        
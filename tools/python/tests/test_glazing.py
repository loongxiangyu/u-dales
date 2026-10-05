from pathlib import Path
from types import SimpleNamespace
from udbase import UDBase
from udprep.udprep_radiation import RadiationSection
from udprep.udprep_glazing import (
    _trace_rays,
    calc_glazing_A_dif_int,
    calc_glazing_TRA_dir,
    calc_glazing_TRA_dif,
    calc_layer_TR_uncoated,
    calc_layer_TR_coated,
    calc_multilayer_TRA,
    calc_multilayer_TRA_raytracing,
)
import tempfile
import unittest
import warnings
import numpy as np
   
   
class Property:
    def __init__(self) -> None:
        self.glaz = {
            30: SimpleNamespace(
                id=30,
                name="Clear",
                T_0=np.array([0.775, 0.775, 0.775]),
                Rf_0=np.array([0.071, 0.071, 0.071]),
                Rb_0=np.array([0.071, 0.071, 0.071]),
                d_g=np.array([0.006, 0.006, 0.006]),
                d_gas=np.array([0.01, 0.01]),
            )
        }


def glazing_type(id, name, T_0, Rf_0, Rb_0):
    """A glazing type with every property _compute_knet writes to aprop_glaz.txt."""
    nlyrs = len(T_0)
    return SimpleNamespace(
        id=id,
        name=name,
        T_0=np.asarray(T_0, dtype=float),
        Rf_0=np.asarray(Rf_0, dtype=float),
        Rb_0=np.asarray(Rb_0, dtype=float),
        d_g=np.full(nlyrs, 0.006),
        emib=np.full(nlyrs, 0.84),
        emif=np.full(nlyrs, 0.84),
        lam_g=np.full(nlyrs, 0.9),
        c_gas=np.full(nlyrs - 1, 1005.0),
        rho_gas=np.full(nlyrs - 1, 1.225),
        lam_gas=np.full(nlyrs - 1, 0.0242),
        d_gas=np.full(nlyrs - 1, 0.01),
        mu_gas=np.full(nlyrs - 1, 1.8e-5),
        z0m=1e-4,
        z0h=1e-6,
    )


def run_compute_knet(glaz, facet_types, nglazlyrs=3, namoptions=None):
    """
    Run the file-writing path of _compute_knet on a scene where every facet
    faces up and no facet sees another, with the direct shortwave stubbed
    out. nglazlyrs is the value the case had before preprocessing, and
    namoptions the text of its namoptions.001 file (none if not given).
    Return the rows of aknet_glaz.txt and aprop_glaz.txt and the namelist
    parameters it saves.
    """
    nfac = len(facet_types)
    saved = {}
    with tempfile.TemporaryDirectory() as tmp:
        if namoptions is not None:
            (Path(tmp) / "namoptions.001").write_text(namoptions, encoding="ascii")
        sim = SimpleNamespace(
            expnr="001",
            nglazlyrs=nglazlyrs,
            nfaclyrs=10,
            path=tmp,
            glaz=glaz,
            facs={"typeid": np.asarray(facet_types)},
            geom=SimpleNamespace(stl=SimpleNamespace(face_normals=np.tile([0.0, 0.0, 1.0], (nfac, 1)))),
            save_param=lambda name, value: saved.__setitem__(name, value),
        )
        solver = RadiationSection("radiation", {"lglaz": True}, sim=sim)
        solver.calc_direct_sw = lambda *args, **kwargs: (np.full(nfac, 300.0), None, None)
        solver._compute_knet(
            np.array([0.0, 0.0, 1.0]), 800.0, 100.0, "facsec", None, True,
            np.full(nfac, 0.2), np.zeros((nfac, nfac)), np.ones(nfac), None,
        )
        aknet = np.loadtxt(Path(tmp) / "aknet_glaz.txt", ndmin=2)
        aprop = np.loadtxt(Path(tmp) / "aprop_glaz.txt", ndmin=2)
    return aknet, aprop, saved

class TestGlazing(unittest.TestCase): 
    def test_glazing_manually_with_original_MATLABcode(self):
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
        knet, knet_glaz, albedo_glaz, solar = solver.calc_knet_glaz(
            sdir, dsky, albedo, vf, svf, phi, facet_types
        )
        print("knet_glaz:", knet_glaz)
        print("albedo_glaz:", albedo_glaz)
        print("knet:", knet)
        self.assertEqual(knet_glaz.shape[0], nglaz)
        self.assertAlmostEqual(albedo_glaz[0][1], Rw_dir, delta=0.0001)
        self.assertAlmostEqual(albedo_glaz[0][2], Rw_dif, delta=0.0001)
        self.assertAlmostEqual(knet_glaz[0][1], knet_layer1, delta=0.0001)

        # The second glazing facet sits at grazing incidence (phi=90 deg). At the
        # Fresnel grazing limit the interface reflects (almost) all incident
        # light, so the direct-beam front reflectance must tend to 1 regardless
        # of the glazing's normal-incidence properties. This is an independent
        # physical sanity check (not tied to any external reference run), and it
        # exercises the second row of albedo_glaz/knet_glaz, which the assertions
        # above never touch.
        self.assertAlmostEqual(albedo_glaz[1][1], 1.0, delta=0.0001)

    def test_single_layer_uncoated_matches_calc_layer_TR_uncoated(self):
        """
        calc_glazing_TRA_dir/calc_glazing_TRA_dif special-case N==1 (a single glazing
        layer): they return calc_layer_TR_uncoated's result directly instead of combining
        layers with calc_multilayer_TRA. test_glazing_properties above always uses 3
        layers, so it never exercises this branch. That is exactly the branch
        which used to return `T_phi.item` (a bound method, missing its call
        parentheses) instead of `T_phi.item()` (a float) and crashed as soon as
        the arithmetic `1 - Tw - Rfw` ran. Calling the N==1 path directly here
        means a future regression of that kind fails this test immediately
        instead of only showing up on a real single-pane case.
        """
        T_0 = np.array([0.775])
        Rf_0 = np.array([0.071])
        Rb_0 = np.array([0.071])  # uncoated: front and back reflectance match
        d_g = np.array([0.006])
        phi = np.deg2rad(45.0)

        Tw, Rfw, Rbw, Aw = calc_glazing_TRA_dir(T_0, Rf_0, Rb_0, d_g, phi)

        # The N==1 branch must hand back plain scalars, not 1-element arrays
        # and not the unbound-method object the earlier bug produced.
        self.assertIsInstance(Tw, float)
        self.assertIsInstance(Rfw, float)
        self.assertIsInstance(Rbw, float)

        # With a single layer, Tw/Rfw/Rbw are exactly calc_layer_TR_uncoated's output for
        # that layer, so we can compare against it directly instead of a
        # hand-copied magic number.
        T_expected, Rf_expected = calc_layer_TR_uncoated(T_0[0], Rf_0[0], phi, d_g[0])
        _, Rb_expected = calc_layer_TR_uncoated(T_0[0], Rb_0[0], phi, d_g[0])
        self.assertAlmostEqual(Tw, T_expected)
        self.assertAlmostEqual(Rfw, Rf_expected)
        self.assertAlmostEqual(Rbw, Rb_expected)
        self.assertAlmostEqual(Aw, 1 - Tw - Rfw)

        TwD, RfwD, RbwD, AwD = calc_glazing_TRA_dif(T_0, Rf_0, Rb_0, d_g)
        self.assertIsInstance(TwD, float)
        self.assertIsInstance(RfwD, float)
        self.assertIsInstance(RbwD, float)
        # The diffuse values integrate calc_layer_TR_uncoated over 0-90 deg internally, so
        # there is no single-call reference to compare against here; we settle
        # for the same absorptance self-consistency check as the direct case.
        self.assertAlmostEqual(AwD, 1 - TwD - RfwD)

    def test_single_layer_coated_matches_calc_layer_TR_coated(self):
        """
        Front and back reflectance differ (a coated pane), which routes
        calc_glazing_TRA_dir/calc_glazing_TRA_dif through calc_layer_TR_coated instead
        of calc_layer_TR_uncoated. test_glazing_properties uses Rf_0 == Rb_0 everywhere,
        so that branch was never covered before this test.
        """
        T_0 = np.array([0.7])  # > 0.645 selects the "clear" coating regression fit
        Rf_0 = np.array([0.1])
        Rb_0 = np.array([0.2])  # coated: front and back reflectance differ
        d_g = np.array([0.006])
        phi = np.deg2rad(30.0)

        Tw, Rfw, Rbw, Aw = calc_glazing_TRA_dir(T_0, Rf_0, Rb_0, d_g, phi)
        T_expected, Rf_expected = calc_layer_TR_coated(T_0[0], Rf_0[0], phi)
        _, Rb_expected = calc_layer_TR_coated(T_0[0], Rb_0[0], phi)
        self.assertAlmostEqual(Tw, T_expected)
        self.assertAlmostEqual(Rfw, Rf_expected)
        self.assertAlmostEqual(Rbw, Rb_expected)
        self.assertAlmostEqual(Aw, 1 - Tw - Rfw)

        TwD, RfwD, RbwD, AwD = calc_glazing_TRA_dif(T_0, Rf_0, Rb_0, d_g)
        self.assertIsInstance(TwD, float)
        self.assertAlmostEqual(AwD, 1 - TwD - RfwD)

    def test_multilayer_matches_reference_and_energy_balance(self):
        """
        N>1 goes through calc_multilayer_TRA, which recursively combines each layer's
        own T/R into the whole-system T/R/A instead of returning calc_layer_TR_uncoated's
        result directly (as the N==1 branch does). Run this for several layer
        counts (edit LAYER_COUNTS to add more) instead of hard-coding a single
        N=3 stack, so a regression in calc_multilayer_TRA's recursion is caught
        regardless of how many panes a real case happens to use. Every layer
        is given the same per-layer properties for simplicity; only the
        number of layers changes between subTests.

        Two checks per layer count:
        - transmitted + front-reflected + sum(absorbed per layer) == 1, an
          energy-balance identity that must hold for any N and any inputs.
          The recursion conserves energy exactly, so the tolerance only
          allows for floating-point round-off. An imbalance of ~1e-4 at
          larger N, once taken for round-off, was a wrong denominator in the
          middle-layer absorptance, which a loose tolerance let through.
        - For N == 3 specifically, Rfw/RfwD also reproduce the MATLAB-derived
          Rw_dir/Rw_dif reference that test_glazing_properties validates
          end-to-end; calling calc_glazing_TRA_dir/dif directly here isolates
          that check from the albedo/view-factor/facet bookkeeping
          calc_knet_glaz also does.
        """
        LAYER_COUNTS = [2, 3, 4, 8, 12]
        Rw_dir_N3 = 0.1608
        Rw_dif_N3 = 0.2398
        phi = np.deg2rad(45.0)

        for N in LAYER_COUNTS:
            with self.subTest(N=N):
                T_0 = np.full(N, 0.775)
                Rf_0 = np.full(N, 0.071)
                Rb_0 = np.full(N, 0.071)
                d_g = np.full(N, 0.006)

                Tw, Rfw, Rbw, Aw = calc_glazing_TRA_dir(T_0, Rf_0, Rb_0, d_g, phi)
                self.assertEqual(Aw.shape, (N,))
                self.assertAlmostEqual(Tw + Rfw + np.sum(Aw), 1.0, delta=1e-12)

                TwD, RfwD, RbwD, AwD = calc_glazing_TRA_dif(T_0, Rf_0, Rb_0, d_g)
                self.assertEqual(AwD.shape, (N,))
                self.assertAlmostEqual(TwD + RfwD + np.sum(AwD), 1.0, delta=1e-12)

                if N == 3:
                    self.assertAlmostEqual(Rfw, Rw_dir_N3, delta=0.0001)
                    self.assertAlmostEqual(RfwD, Rw_dif_N3, delta=0.0001)

    def test_facet_classification_ignores_non_glazing_types(self):
        """
        calc_knet_glaz decides which facets are glazing by comparing
        facet_types against the ids in self.glaz (udprep_radiation.py's glaz_idx/nglaz),
        then only those facets get marked specular and given the glazing
        albedo (Rf_sys_dif overwrites albedo[j] for glazing facets only).
        test_glazing_properties only ever uses facet_types=[30, 1, 30], so it
        never proves that an arbitrary *other* type (not just "1") is also
        left alone, or that non-glazing facets keep whatever albedo the
        caller passed in. Use two different non-glazing types (1 and 7) here
        to check both are treated identically -- i.e. classification really
        is "matches a glazing id" and not e.g. "not equal to 1".
        """
        svf = np.array([1.0, 1.0, 1.0, 1.0])
        phi = np.deg2rad(np.array([45, 45, 0, 0]))
        facet_types = np.array([30, 1, 7, 30])  # 30 = glazing, 1 and 7 = non-glazing
        sdir = np.cos(phi) * np.array([240.51, 240.51, 240.51, 240.51])
        dsky = 246.73
        original_albedo = np.array([0.2, 0.35, 0.5, 0.2])
        albedo = original_albedo.copy()
        vf = np.zeros((4, 4))

        sim = Property()
        glaz = sim.glaz[30]
        solver = RadiationSection("radiation", {}, sim=sim)
        nglaz = np.sum(facet_types == glaz.id)
        knet, knet_glaz, albedo_glaz, solar = solver.calc_knet_glaz(
            sdir, dsky, albedo, vf, svf, phi, facet_types
        )

        self.assertEqual(nglaz, 2)
        self.assertEqual(knet_glaz.shape[0], nglaz)
        # The first column is the facet index; it must list exactly the
        # facets whose type matches glaz.id, in ascending order, and nothing
        # from facets 1 or 2.
        np.testing.assert_array_equal(knet_glaz[:, 0], [0, 3])
        np.testing.assert_array_equal(albedo_glaz[:, 0], [0, 3])

        # albedo is mutated in place: glazing facets (0, 3) get overwritten
        # with the glazing's diffuse front reflectance (albedo_glaz[:, 2])
        # plus the diffuse light returned from the room (room albedo 0.2);
        # non-glazing facets (1, 2) -- regardless of which non-glazing type
        # they are -- must keep the caller's original albedo untouched.
        T_dif, _, Rb_dif, _ = calc_glazing_TRA_dif(glaz.T_0, glaz.Rf_0, glaz.Rb_0, glaz.d_g)
        room_return = T_dif * 0.2 / (1 - Rb_dif * 0.2) * T_dif
        np.testing.assert_allclose(albedo[[0, 3]], albedo_glaz[:, 2] + room_return)
        np.testing.assert_array_equal(albedo[[1, 2]], original_albedo[[1, 2]])

    def test_incidence_angle_monotonic_and_recovers_normal_incidence(self):
        """
        calc_glazing_TRA_dir takes phi (incidence angle) and feeds it through
        calc_layer_TR_uncoated's Fresnel reflectance formula. Every other test in this
        file only ever calls it at a single fixed angle (45 or 90 deg), so a
        regression that scrambled the angle dependence (e.g. swapped sin/cos,
        or broke the phi_prime refraction angle) would not be caught. Check
        two model-independent physical properties instead of a hard-coded
        curve of numbers:
        - at normal incidence (phi=0) the whole angle-dependent machinery
          must reduce to the normal-incidence inputs T_0/Rf_0/Rb_0 themselves;
        - as phi sweeps from 0 to 89 deg, transmittance must decrease
          monotonically and front reflectance must trend upward overall (more
          of the beam is reflected at grazing angles). Rfw is allowed a tiny
          (<1e-4) dip between consecutive angles near normal incidence: this
          model combines interface reflection with path-length-dependent
          absorption, so Rfw is not perfectly monotonic step-to-step there
          (verified: the observed dip from 0 to 15 deg is ~7e-6) even though
          the overall trend and the endpoints are not in question.
        """
        T_0 = np.array([0.775])
        Rf_0 = np.array([0.071])
        Rb_0 = np.array([0.071])
        d_g = np.array([0.006])

        Tw0, Rfw0, Rbw0, _ = calc_glazing_TRA_dir(T_0, Rf_0, Rb_0, d_g, phi=0.0)
        self.assertAlmostEqual(Tw0, T_0[0])
        self.assertAlmostEqual(Rfw0, Rf_0[0])
        self.assertAlmostEqual(Rbw0, Rb_0[0])

        angles_deg = [0, 15, 30, 45, 60, 75, 89]
        Tw_prev, Rfw_prev = None, None
        for deg in angles_deg:
            with self.subTest(angle_deg=deg):
                Tw, Rfw, Rbw, _ = calc_glazing_TRA_dir(
                    T_0, Rf_0, Rb_0, d_g, np.deg2rad(deg)
                )
                self.assertGreaterEqual(Tw, 0.0)
                self.assertLessEqual(Rfw, 1.0)
                if Tw_prev is not None:
                    self.assertLess(Tw, Tw_prev)
                    self.assertGreaterEqual(Rfw, Rfw_prev - 1e-4)
                Tw_prev, Rfw_prev = Tw, Rfw

        # The overall trend across the full sweep is unambiguous regardless
        # of the small step-to-step dip checked above.
        self.assertGreater(Rfw_prev, Rfw0)

    def test_diffuse_hemispherical_average_is_correct(self):
        """
        calc_glazing_TRA_dif computes the "diffuse" (hemispherical) optical
        properties by averaging calc_layer_TR_uncoated over incidence angles 0-90 deg,
        weighted by 2*sin(phi)*cos(phi) -- the standard Lambertian/
        hemispherical weighting (its own integral over 0..pi/2 is exactly 1,
        i.e. it is a normalized probability density over incidence angle for
        diffuse radiation). None of the other tests in this file check that
        integration itself: test_multilayer_... and test_glazing_properties
        only compare the *result* against a MATLAB reference for one fixed
        3-layer, 45 deg stack, so a bug in the integration that happened to
        still match that single number would slip through. This test instead
        checks two things that don't depend on any external reference:

        - recompute the same hemispherical average by hand from calc_layer_TR_uncoated
          (the already-validated angle-dependent primitive) for a single
          layer, so no calc_multilayer_TRA layer combination is involved, and check
          calc_glazing_TRA_dif reproduces it exactly;
        - since transmittance only decreases and reflectance only increases
          as the incidence angle grows (see test_incidence_angle_...), their
          angle-averaged ("diffuse") values must fall strictly between the
          normal-incidence (phi=0) value and the grazing-incidence
          (phi->90 deg) value.
        """
        T_0 = np.array([0.775])
        Rf_0 = np.array([0.071])
        Rb_0 = np.array([0.071])
        d_g = np.array([0.006])

        TwD, RfwD, RbwD, AwD = calc_glazing_TRA_dif(T_0, Rf_0, Rb_0, d_g)

        # Manual reference: integrate calc_layer_TR_uncoated over 0-90 deg by hand,
        # using the same normalized Lambertian weighting.
        deg = np.arange(0, 91, 1)
        phi = np.deg2rad(deg)
        weight = 2 * np.cos(phi) * np.sin(phi)
        self.assertAlmostEqual(np.trapezoid(weight, phi), 1.0, delta=1e-3)

        T = np.empty(len(deg))
        Rf = np.empty(len(deg))
        for j, p in enumerate(phi):
            T[j], Rf[j] = calc_layer_TR_uncoated(T_0[0], Rf_0[0], p, d_g[0])
        TwD_expected = np.trapezoid(T * weight, phi)
        RfwD_expected = np.trapezoid(Rf * weight, phi)
        self.assertAlmostEqual(TwD, TwD_expected)
        self.assertAlmostEqual(RfwD, RfwD_expected)

        # Physical bound, independent of the manual integration above.
        Tw0, Rfw0, _, _ = calc_glazing_TRA_dir(T_0, Rf_0, Rb_0, d_g, phi=0.0)
        Tw90, Rfw90, _, _ = calc_glazing_TRA_dir(
            T_0, Rf_0, Rb_0, d_g, phi=np.deg2rad(89.9)
        )
        self.assertTrue(Tw90 < TwD < Tw0)
        self.assertTrue(Rfw0 < RfwD < Rfw90)

    def test_multilayer_TRA_matches_raytracing(self):
        """
        calc_multilayer_TRA combines the layers with closed-form recursive
        formulas; calc_multilayer_TRA_raytracing (Python version of
        WindowRayTracing_fixed.m) gets the same quantities by splitting rays
        at every layer until the energy left untraced is below 1e-12. The two
        share no formulas, so their agreement checks the recursion
        independently. The stacks have different layers and coated layers
        (Rf != Rb): with identical, uncoated layers, as in the other tests
        here, several mistakes in the recursion give the right answer anyway.
        """
        stacks = {
            "clear_3": ([0.775] * 3, [0.071] * 3, [0.071] * 3),
            "low_e_3": ([0.6, 0.775, 0.775], [0.17, 0.071, 0.071], [0.22, 0.071, 0.071]),
            "coated_2": ([0.6, 0.5], [0.10, 0.30], [0.25, 0.05]),
            "coated_3": ([0.6, 0.5, 0.7], [0.10, 0.30, 0.08], [0.25, 0.05, 0.20]),
            "mixed_4": ([0.775, 0.6, 0.7, 0.5], [0.071, 0.15, 0.10, 0.25], [0.071, 0.05, 0.10, 0.12]),
        }
        rng = np.random.default_rng(0)
        for k in range(10):
            N = int(rng.integers(2, 7))
            stacks[f"random_{k}"] = (
                rng.uniform(0.3, 0.8, N), rng.uniform(0.02, 0.2, N), rng.uniform(0.02, 0.2, N)
            )

        for name, (T, Rf, Rb) in stacks.items():
            with self.subTest(stack=name):
                T, Rf, Rb = np.asarray(T), np.asarray(Rf), np.asarray(Rb)
                T_sys, Rf_sys, Rb_sys, A_sys_lyrs = calc_multilayer_TRA(T, Rf, Rb)
                T_ray, Rf_ray, Rb_ray, A_ray = calc_multilayer_TRA_raytracing(T, Rf, Rb)
                self.assertAlmostEqual(T_sys, T_ray, delta=1e-10)
                self.assertAlmostEqual(Rf_sys, Rf_ray, delta=1e-10)
                self.assertAlmostEqual(Rb_sys, Rb_ray, delta=1e-10)
                np.testing.assert_allclose(A_sys_lyrs, A_ray, rtol=0, atol=1e-10)

    def test_interior_incidence_absorptance(self):
        """
        Light reflected by the room hits the glazing on its back face.
        calc_glazing_A_dif_int gets the layer absorptance for it by flipping
        the stack, which must also swap the front and back faces of every
        layer and flip the result back to exterior-to-interior order.
        Forgetting the face swap computes a different glazing, but only for
        coated layers (Rf != Rb), which the other tests here hardly use.
        Two checks on coated stacks:
        - energy balance for light from the interior, with T_sys and Rb_sys
          from the unflipped stack: T_sys + Rb_sys + sum(A_int) == 1;
        - layer by layer, A_int equals ray tracing from the interior on the
          unflipped stack's per-layer diffuse properties, which also catches
          a wrong layer order (the sum alone does not).
        """
        stacks = {
            "low_e_1": ([0.6], [0.17], [0.22]),
            "low_e_3": ([0.6, 0.775, 0.775], [0.17, 0.071, 0.071], [0.22, 0.071, 0.071]),
            "coated_3": ([0.6, 0.5, 0.7], [0.10, 0.30, 0.08], [0.25, 0.05, 0.20]),
        }
        for name, (T_0, Rf_0, Rb_0) in stacks.items():
            with self.subTest(stack=name):
                T_0, Rf_0, Rb_0 = np.asarray(T_0), np.asarray(Rf_0), np.asarray(Rb_0)
                d_g = np.full(len(T_0), 0.006)
                A_int = np.atleast_1d(calc_glazing_A_dif_int(T_0, Rf_0, Rb_0, d_g))

                T_sys, _, Rb_sys, _ = calc_glazing_TRA_dif(T_0, Rf_0, Rb_0, d_g)
                self.assertAlmostEqual(T_sys + Rb_sys + np.sum(A_int), 1.0, delta=1e-12)

                # per-layer diffuse properties: each layer on its own is a 1-layer stack
                lyrs = np.array([
                    calc_glazing_TRA_dif(T_0[k:k + 1], Rf_0[k:k + 1], Rb_0[k:k + 1], d_g[k:k + 1])[:3]
                    for k in range(len(T_0))
                ])
                _, _, A_ray = _trace_rays(lyrs[:, 0], lyrs[:, 1], lyrs[:, 2], False, 1e-12, 100000)
                np.testing.assert_allclose(A_int, A_ray, rtol=0, atol=1e-10)

    def test_glazing_facet_conserves_shortwave_with_room(self):
        """
        Shortwave on a glazing facet is reflected by the glazing, absorbed in
        its layers, or transmitted into the room. The room (albedo 0.2)
        reflects it back and forth with the glazing; on each pass the glazing
        absorbs part of it, reflects part back into the room and lets
        T_sys_dif pass to the exterior. So incident = leaving the facet
        (glazing reflection + light returned from the room) + absorbed in the
        layers + absorbed in the room, where the room absorbs
        (1 - 0.2) * K_trans / (1 - 0.2 * Rb_sys_dif). A black receiver that
        sees only the glazing collects everything leaving the facet. The
        balance fails if the room-returned light is dropped (the direct or
        the diffuse part) or counted twice. Coated layers make Rf != Rb.
        """
        sim = SimpleNamespace(glaz={30: SimpleNamespace(
            id=30,
            name="LowE",
            T_0=np.array([0.6, 0.775, 0.775]),
            Rf_0=np.array([0.17, 0.071, 0.071]),
            Rb_0=np.array([0.22, 0.071, 0.071]),
            d_g=np.full(3, 0.006),
        )})
        albedo_room = 0.2  # as in calc_knet_glaz
        phi = np.deg2rad(np.array([45.0, 0.0]))
        facet_types = np.array([30, 1])  # glazing, black receiver
        sdir = np.array([600.0 * np.cos(phi[0]), 0.0])
        dsky = 150.0
        svf = np.array([1.0, 0.0])  # only the glazing sees the sky
        vf = np.array([[0.0, 0.0], [1.0, 0.0]])  # the receiver sees only the glazing
        albedo = np.array([0.2, 0.0])

        solver = RadiationSection("radiation", {}, sim=sim)
        knet, knet_glaz, _, _ = solver.calc_knet_glaz(sdir, dsky, albedo, vf, svf, phi, facet_types)

        glaz = sim.glaz[30]
        T_dir, _, _, _ = calc_glazing_TRA_dir(glaz.T_0, glaz.Rf_0, glaz.Rb_0, glaz.d_g, phi[0])
        T_dif, _, Rb_dif, _ = calc_glazing_TRA_dif(glaz.T_0, glaz.Rf_0, glaz.Rb_0, glaz.d_g)
        K_trans = T_dir * sdir[0] + T_dif * dsky
        absorbed_room = (1 - albedo_room) * K_trans / (1 - albedo_room * Rb_dif)

        incident = sdir[0] + dsky
        leaving = knet[1]  # all absorbed by the black receiver
        absorbed_layers = np.sum(knet_glaz[0, 1:])
        self.assertAlmostEqual(leaving + absorbed_layers + absorbed_room, incident, delta=1e-9)

    def test_aknet_glaz_lists_1_based_facet_numbers(self):
        """
        Python indexes facets from 0, while the Fortran solver numbers them
        from 1 (row n of facets.inp is facet n) and looks up each glazing
        facet in aknet_glaz.txt by that number. calc_knet_glaz returns 0-based
        indices, so _compute_knet must add 1 when it writes the file: a
        0-based index makes every glazing facet miss its row and read
        netsw_glaz(0,:) out of bounds in Fortran. Run the file-writing path of
        _compute_knet on a small scene, with the direct shortwave stubbed out,
        and check the first column holds the 1-based facet numbers.
        """
        facet_types = np.array([1, 30, 1, 30, 30, 7])  # glazing at 0-based 1, 3, 4
        clear = glazing_type(30, "Clear", [0.775] * 3, [0.071] * 3, [0.071] * 3)
        rows, _, saved = run_compute_knet({30: clear}, facet_types)

        np.testing.assert_array_equal(rows[:, 0], [2, 4, 5])
        self.assertEqual(rows.shape[1], 1 + 2 * 3)  # facet number + two surfaces per layer
        self.assertEqual(saved["nglaz"], 3)

    def test_each_glazing_type_uses_its_own_properties(self):
        """
        With several glazing types, calc_knet_glaz must give every glazing
        facet the optical properties of its own type. No facet sees another
        here, so a glazing facet only depends on its own type, sun and sky:
        in a scene that mixes two types it must get exactly what it gets when
        its type is the only glazing type. Facets 0 and 1 have the same sun
        and incidence angle, so their results differ only through the type.
        """
        clear = glazing_type(30, "Clear", [0.775] * 3, [0.071] * 3, [0.071] * 3)
        lowe = glazing_type(31, "LowE", [0.6, 0.775, 0.775], [0.17, 0.071, 0.071], [0.22, 0.071, 0.071])
        facet_types = np.array([30, 31, 1, 31, 30])
        phi = np.deg2rad(np.array([40.0, 40.0, 40.0, 70.0, 10.0]))
        sdir = 600.0 * np.cos(phi)
        dsky = 150.0
        svf = np.ones(5)
        vf = np.zeros((5, 5))

        def run(glaz):
            solver = RadiationSection("radiation", {}, sim=SimpleNamespace(glaz=glaz))
            albedo = np.full(5, 0.3)
            _, knet_glaz, albedo_glaz, _ = solver.calc_knet_glaz(sdir, dsky, albedo, vf, svf, phi, facet_types)
            return knet_glaz, albedo_glaz, albedo

        knet_glaz, albedo_glaz, albedo = run({30: clear, 31: lowe})
        np.testing.assert_array_equal(knet_glaz[:, 0], [0, 1, 3, 4])

        for glaz, rows in ((clear, [0, 3]), (lowe, [1, 2])):  # rows of the facets of this type
            with self.subTest(glazing=glaz.name):
                knet_glaz_1, albedo_glaz_1, albedo_1 = run({glaz.id: glaz})
                facets = knet_glaz[rows, 0].astype(int)
                np.testing.assert_allclose(knet_glaz[rows], knet_glaz_1, rtol=0, atol=1e-12)
                np.testing.assert_allclose(albedo_glaz[rows], albedo_glaz_1, rtol=0, atol=1e-12)
                np.testing.assert_allclose(albedo[facets], albedo_1[facets], rtol=0, atol=1e-12)

        self.assertFalse(np.allclose(knet_glaz[0, 1:], knet_glaz[1, 1:]))
        self.assertNotAlmostEqual(albedo[0], albedo[1])
        self.assertEqual(albedo[2], 0.3)  # the non-glazing facet keeps its albedo

    def test_aprop_glaz_has_one_row_per_glazing_type(self):
        """
        The solver reads nglaztypes rows from aprop_glaz.txt and finds the
        properties of a glazing facet by its type id in the first column.
        Every glazing type must give one row with its own properties in the
        column order the solver reads (id, emib, emif, lam_g, d_g, c_gas,
        rho_gas, lam_gas, d_gas, mu_gas, z0m, z0h), and nglaztypes must be
        saved for the solver. aknet_glaz.txt lists the facets of all types.
        """
        clear = glazing_type(30, "Clear", [0.775] * 3, [0.071] * 3, [0.071] * 3)
        lowe = glazing_type(31, "LowE", [0.6, 0.775, 0.775], [0.17, 0.071, 0.071], [0.22, 0.071, 0.071])
        lowe.emif = np.array([0.1, 0.84, 0.84])  # low-e coating on the outer face
        lowe.lam_g = np.array([0.8, 0.9, 1.0])
        lowe.d_gas = np.array([0.012, 0.016])
        lowe.mu_gas = np.array([1.7e-5, 2.2e-5])
        lowe.z0m = 2e-4
        facet_types = np.array([31, 1, 30, 31])
        aknet, aprop, saved = run_compute_knet({30: clear, 31: lowe}, facet_types)

        np.testing.assert_array_equal(aknet[:, 0], [1, 3, 4])
        self.assertEqual(saved["nglaz"], 3)
        self.assertEqual(saved["nglaztypes"], 2)
        self.assertEqual(aprop.shape, (2, 1 + 4 * 3 + 5 * 2 + 2))  # 3 layers, 2 gaps
        for row, glaz in zip(aprop, (clear, lowe)):
            with self.subTest(glazing=glaz.name):
                expected = np.concatenate((
                    [glaz.id], glaz.emib, glaz.emif, glaz.lam_g, glaz.d_g,
                    glaz.c_gas, glaz.rho_gas, glaz.lam_gas, glaz.d_gas, glaz.mu_gas,
                    [glaz.z0m, glaz.z0h],
                ))
                np.testing.assert_allclose(row, expected, rtol=1e-6)

    def test_glazing_types_must_have_the_same_layers(self):
        """
        All glazing types share nglazlyrs: the solver reads 2*nglazlyrs
        absorbed values per glazing facet, and nglazlyrs values of each layer
        property (nglazlyrs-1 of each gap property) per glazing type. A type
        with another number of layers, or a property with the wrong number
        of values, would shift the columns the solver reads, so
        _compute_knet must stop with an error that names the glazing type.
        """
        clear = glazing_type(30, "Clear", [0.775] * 3, [0.071] * 3, [0.071] * 3)
        double = glazing_type(31, "Double", [0.775] * 2, [0.071] * 2, [0.071] * 2)
        with self.assertRaisesRegex(ValueError, r"\[Double\] in material.toml has 2 glazing layers and \[Clear\] has 3"):
            run_compute_knet({30: clear, 31: double}, np.array([30, 31]))

        clear.mu_gas = np.full(3, 1.8e-5)  # one value per layer instead of per gap
        with self.assertRaisesRegex(ValueError, r"mu_gas of \[Clear\] in material.toml has 3 values, expected 2"):
            run_compute_knet({30: clear}, np.array([30]))

        clear = glazing_type(30, "Clear", [0.775] * 3, [0.071] * 3, [0.071] * 3)
        del clear.z0h
        with self.assertRaisesRegex(ValueError, r"\[Clear\] in material.toml has no z0h"):
            run_compute_knet({30: clear}, np.array([30]))

    def test_material_toml_tables_are_glazing_types(self):
        """
        Every top-level table of material.toml is one glazing type, named
        after the table (e.g. [Clear], [LowE]) and linked to the facets by its
        id, and UDBase keys the types by id. A file with the old single
        [Glaz] table is one glazing type named Glaz. Two tables with the same
        id, a table without id, a top-level value that is not a table, or a
        file without any table are errors.
        """
        namoptions = "\n".join([
            "&DOMAIN", " itot = 4", " jtot = 3", " ktot = 2",
            " xlen = 40.0", " ylen = 30.0", " zsize = 20.0", "/",
            "&ENERGYBALANCE", " lglaz = .true.", "/",
        ]) + "\n"

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "namoptions.001").write_text(namoptions, encoding="ascii")

            def load(toml):
                (path / "material.toml").write_text(toml, encoding="ascii")
                return UDBase("1", path, load_geometry=False, suppress_load_warnings=True)

            sim = load("[Clear]\nid = 30\nT_0 = [0.775, 0.775]\n\n[LowE]\nid = 31\nT_0 = [0.6, 0.775]\n")
            self.assertEqual(list(sim.glaz), [30, 31])
            self.assertEqual([glaz.name for glaz in sim.glaz.values()], ["Clear", "LowE"])
            self.assertEqual(sim.glaz[31].T_0, [0.6, 0.775])

            sim = load("[Glaz]\nid = 30\nT_0 = [0.775]\n")
            self.assertEqual(list(sim.glaz), [30])
            self.assertEqual(sim.glaz[30].name, "Glaz")

            for toml, message in (
                ("[Clear]\nid = 30\n\n[LowE]\nid = 30\n", r"\[LowE\] and \[Clear\] have the same id 30"),
                ("[Clear]\nT_0 = [0.775]\n", r"\[Clear\] has no id"),
                ("nglazlyrs = 2\n\n[Clear]\nid = 30\n", r"'nglazlyrs' must be a table"),
                ("", r"defines no glazing type"),
            ):
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    load(toml)

    def test_glazing_knet_is_shortwave_absorbed_by_its_layers(self):
        """
        knet goes to netsw.inp and the solver outputs it as netsw. For glazing
        facets the solver uses the absorbed shortwave of each glazing surface
        from aknet_glaz.txt instead, so their knet must be what those surfaces
        absorb in total: the sum of the facet's row in knet_glaz, direct
        shortwave included. The reflection loop alone gives a glazing facet
        only the diffuse shortwave absorbed by its layers and the room behind
        it. Non-glazing facets keep the net shortwave of the reflection loop.
        """
        clear = glazing_type(30, "Clear", [0.775] * 3, [0.071] * 3, [0.071] * 3)
        lowe = glazing_type(31, "LowE", [0.6, 0.775, 0.775], [0.17, 0.071, 0.071], [0.22, 0.071, 0.071])
        facet_types = np.array([30, 1, 31, 7, 30])
        phi = np.deg2rad(np.array([30.0, 50.0, 40.0, 60.0, 70.0]))
        sdir = 600.0 * np.cos(phi)
        sdir[2] = 0.0  # a shaded glazing facet
        dsky = 100.0
        vf = np.array([  # equal areas, so the view factors are symmetric
            [0.0, 0.2, 0.1, 0.1, 0.0],
            [0.2, 0.0, 0.2, 0.1, 0.1],
            [0.1, 0.2, 0.0, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.0, 0.2],
            [0.0, 0.1, 0.1, 0.2, 0.0],
        ])
        svf = 1.0 - vf.sum(axis=1)
        albedo = np.array([0.3, 0.2, 0.3, 0.4, 0.3])

        solver = RadiationSection("radiation", {}, sim=SimpleNamespace(glaz={30: clear, 31: lowe}))
        loop = {}
        reflections = solver.calc_reflections_sw_glaz

        def keep_loop_knet(*args, **kwargs):
            loop["knet"], kin = reflections(*args, **kwargs)
            return loop["knet"].copy(), kin

        solver.calc_reflections_sw_glaz = keep_loop_knet
        knet, knet_glaz, _, _ = solver.calc_knet_glaz(sdir, dsky, albedo, vf, svf, phi, facet_types)

        glazing = knet_glaz[:, 0].astype(int)
        np.testing.assert_array_equal(glazing, [0, 2, 4])
        np.testing.assert_allclose(knet[glazing], knet_glaz[:, 1:].sum(axis=1), rtol=1e-12)
        others = np.setdiff1d(np.arange(len(facet_types)), glazing)
        np.testing.assert_array_equal(knet[others], loop["knet"][others])

    def test_glazing_requires_surface_energy_balance(self):
        """
        Glazing is solved in the surface energy balance. Without lEB the
        shortwave step neither computes reflections nor the glazing, and the
        albedo of a glazing type missing from factypes is NaN and would end up
        in netsw. _compute_knet must stop before any of that.
        """
        clear = glazing_type(30, "Clear", [0.775] * 3, [0.071] * 3, [0.071] * 3)
        solver = RadiationSection("radiation", {"lglaz": True}, sim=SimpleNamespace(glaz={30: clear}))
        with self.assertRaisesRegex(ValueError, r"lglaz = \.true\. requires lEB = \.true\."):
            solver._compute_knet(
                np.array([0.0, 0.0, 1.0]), 800.0, 100.0, "facsec", None, False,
                np.full(2, 0.2), None, None, np.ones(2),
            )

    def test_nglazlyrs_is_written_from_material_toml(self):
        """
        The number of glazing layers comes from material.toml, so preprocessing
        writes nglazlyrs to the namoptions file for the solver, like nglaz and
        nglaztypes. When the namoptions file sets a different value by hand,
        that value is replaced with a warning; when it does not set nglazlyrs
        at all (the default 1 applies), it is written without a warning.
        """
        double = glazing_type(30, "Double", [0.775] * 2, [0.071] * 2, [0.071] * 2)

        with self.assertWarnsRegex(UserWarning, r"nglazlyrs = 3 in the namoptions file is replaced by 2"):
            aknet, aprop, saved = run_compute_knet(
                {30: double}, np.array([30, 1]), nglazlyrs=3,
                namoptions="&ENERGYBALANCE\n lglaz = .true.\n nglazlyrs = 3\n/\n",
            )
        self.assertEqual(saved["nglazlyrs"], 2)
        self.assertEqual(aknet.shape[1], 1 + 2 * 2)  # facet number + two surfaces per layer
        self.assertEqual(aprop.shape[1], 1 + 4 * 2 + 5 * 1 + 2)  # 2 layers, 1 gap

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, _, saved = run_compute_knet(
                {30: double}, np.array([30, 1]), nglazlyrs=1,
                namoptions="&ENERGYBALANCE\n lglaz = .true.\n/\n",
            )
        self.assertEqual(saved["nglazlyrs"], 2)
        self.assertEqual([str(w.message) for w in caught if "nglazlyrs" in str(w.message)], [])

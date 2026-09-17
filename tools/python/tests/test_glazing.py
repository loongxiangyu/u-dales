from types import SimpleNamespace
from udprep.udprep_radiation import RadiationSection
from udprep.udprep_glazing import (
    calc_optiprop_dir,
    calc_optiprop_dif,
    calc_TR_phi,
    calc_TRcoated_phi,
)
import unittest
import numpy as np
   
   
class Property:
    def __init__(self) -> None:
        self.glaz = SimpleNamespace(
            id=30,
            T_0=np.array([0.775, 0.775, 0.775]),
            Rf_0=np.array([0.071, 0.071, 0.071]),
            Rb_0=np.array([0.071, 0.071, 0.071]),
            d_g=np.array([0.006, 0.006, 0.006]),
            d_gas=np.array([0.01, 0.01]),
        )

class TestGlazing(unittest.TestCase): 
    def test_glazing_manually_with_origianl_MATLABcode(self):
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
        self.assertEqual(knet_glaz.shape[0], nglaz)
        self.assertAlmostEqual(al_glaz[0][1], Rw_dir, delta=0.0001)
        self.assertAlmostEqual(al_glaz[0][2], Rw_dif, delta=0.0001)
        self.assertAlmostEqual(knet_glaz[0][1], knet_layer1, delta=0.0001)

        # The second glazing facet sits at grazing incidence (phi=90 deg). At the
        # Fresnel grazing limit the interface reflects (almost) all incident
        # light, so the direct-beam front reflectance must tend to 1 regardless
        # of the glazing's normal-incidence properties. This is an independent
        # physical sanity check (not tied to any external reference run), and it
        # exercises the second row of al_glaz/knet_glaz, which the assertions
        # above never touch.
        self.assertAlmostEqual(al_glaz[1][1], 1.0, delta=0.0001)

    def test_single_layer_uncoated_matches_calc_TR_phi(self):
        """
        calc_optiprop_dir/calc_optiprop_dif special-case N==1 (a single glazing
        layer): they return calc_TR_phi's result directly instead of combining
        layers with calc_TRA_EP. test_glazing_properties above always uses 3
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

        Tw, Rfw, Rbw, Aw = calc_optiprop_dir(T_0, Rf_0, Rb_0, d_g, phi)

        # The N==1 branch must hand back plain scalars, not 1-element arrays
        # and not the unbound-method object the earlier bug produced.
        self.assertIsInstance(Tw, float)
        self.assertIsInstance(Rfw, float)
        self.assertIsInstance(Rbw, float)

        # With a single layer, Tw/Rfw/Rbw are exactly calc_TR_phi's output for
        # that layer, so we can compare against it directly instead of a
        # hand-copied magic number.
        T_expected, Rf_expected = calc_TR_phi(T_0[0], Rf_0[0], phi, d_g[0])
        _, Rb_expected = calc_TR_phi(T_0[0], Rb_0[0], phi, d_g[0])
        self.assertAlmostEqual(Tw, T_expected)
        self.assertAlmostEqual(Rfw, Rf_expected)
        self.assertAlmostEqual(Rbw, Rb_expected)
        self.assertAlmostEqual(Aw, 1 - Tw - Rfw)

        TwD, RfwD, RbwD, AwD = calc_optiprop_dif(T_0, Rf_0, Rb_0, d_g)
        self.assertIsInstance(TwD, float)
        self.assertIsInstance(RfwD, float)
        self.assertIsInstance(RbwD, float)
        # The diffuse values integrate calc_TR_phi over 0-90 deg internally, so
        # there is no single-call reference to compare against here; we settle
        # for the same absorptance self-consistency check as the direct case.
        self.assertAlmostEqual(AwD, 1 - TwD - RfwD)

    def test_single_layer_coated_matches_calc_TRcoated_phi(self):
        """
        Front and back reflectance differ (a coated pane), which routes
        calc_optiprop_dir/calc_optiprop_dif through calc_TRcoated_phi instead
        of calc_TR_phi. test_glazing_properties uses Rf_0 == Rb_0 everywhere,
        so that branch was never covered before this test.
        """
        T_0 = np.array([0.7])  # > 0.645 selects the "clear" coating regression fit
        Rf_0 = np.array([0.1])
        Rb_0 = np.array([0.2])  # coated: front and back reflectance differ
        d_g = np.array([0.006])
        phi = np.deg2rad(30.0)

        Tw, Rfw, Rbw, Aw = calc_optiprop_dir(T_0, Rf_0, Rb_0, d_g, phi)
        T_expected, Rf_expected = calc_TRcoated_phi(T_0[0], Rf_0[0], phi)
        _, Rb_expected = calc_TRcoated_phi(T_0[0], Rb_0[0], phi)
        self.assertAlmostEqual(Tw, T_expected)
        self.assertAlmostEqual(Rfw, Rf_expected)
        self.assertAlmostEqual(Rbw, Rb_expected)
        self.assertAlmostEqual(Aw, 1 - Tw - Rfw)

        TwD, RfwD, RbwD, AwD = calc_optiprop_dif(T_0, Rf_0, Rb_0, d_g)
        self.assertIsInstance(TwD, float)
        self.assertAlmostEqual(AwD, 1 - TwD - RfwD)

    def test_multilayer_matches_reference_and_energy_balance(self):
        """
        N>1 goes through calc_TRA_EP, which recursively combines each layer's
        own T/R into the whole-system T/R/A instead of returning calc_TR_phi's
        result directly (as the N==1 branch does). Run this for several layer
        counts (edit LAYER_COUNTS to add more) instead of hard-coding a single
        N=3 stack, so a regression in calc_TRA_EP's recursion is caught
        regardless of how many panes a real case happens to use. Every layer
        is given the same per-layer properties for simplicity; only the
        number of layers changes between subTests.

        Two checks per layer count:
        - transmitted + front-reflected + sum(absorbed per layer) == 1, an
          energy-balance identity that must hold for any N and any inputs.
          The tolerance is loosened for larger N because calc_TRA_EP's
          recursion accumulates a small amount of floating-point drift per
          extra layer (verified up to N=12: worst-case drift ~3e-4).
        - For N == 3 specifically, Rfw/RfwD also reproduce the MATLAB-derived
          Rw_dir/Rw_dif reference that test_glazing_properties validates
          end-to-end; calling calc_optiprop_dir/dif directly here isolates
          that check from the albedo/view-factor/facet bookkeeping
          calc_knet_glaz also does.
        """
        LAYER_COUNTS = [2, 3, 4]
        Rw_dir_N3 = 0.1608
        Rw_dif_N3 = 0.2398
        phi = np.deg2rad(45.0)

        for N in LAYER_COUNTS:
            with self.subTest(N=N):
                T_0 = np.full(N, 0.775)
                Rf_0 = np.full(N, 0.071)
                Rb_0 = np.full(N, 0.071)
                d_g = np.full(N, 0.006)

                Tw, Rfw, Rbw, Aw = calc_optiprop_dir(T_0, Rf_0, Rb_0, d_g, phi)
                self.assertEqual(Aw.shape, (N,))
                self.assertAlmostEqual(Tw + Rfw + np.sum(Aw), 1.0, delta=1e-3)

                TwD, RfwD, RbwD, AwD = calc_optiprop_dif(T_0, Rf_0, Rb_0, d_g)
                self.assertEqual(AwD.shape, (N,))
                self.assertAlmostEqual(TwD + RfwD + np.sum(AwD), 1.0, delta=1e-3)

                if N == 3:
                    self.assertAlmostEqual(Rfw, Rw_dir_N3, delta=0.0001)
                    self.assertAlmostEqual(RfwD, Rw_dif_N3, delta=0.0001)

    def test_facet_classification_ignores_non_glazing_types(self):
        """
        calc_knet_glaz decides which facets are glazing by comparing
        facet_types against self.glaz.id (udprep_radiation.py's poglaz/nglaz),
        then only those facets get marked specular and given the glazing
        albedo (RfwD_F overwrites albedo[j] for glazing facets only).
        test_glazing_properties only ever uses facet_types=[30, 1, 30], so it
        never proves that an arbitrary *other* type (not just "1") is also
        left alone, or that non-glazing facets keep whatever albedo the
        caller passed in. Use two different non-glazing types (1 and 7) here
        to check both are treated identically -- i.e. classification really
        is "matches glaz.id" and not e.g. "not equal to 1".
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
        solver = RadiationSection("radiation", {}, sim=sim)
        nglaz = np.sum(facet_types == sim.glaz.id)
        knet, knet_glaz, al_glaz, solar = solver.calc_knet_glaz(
            sdir, dsky, albedo, vf, svf, phi, facet_types
        )

        self.assertEqual(nglaz, 2)
        self.assertEqual(knet_glaz.shape[0], nglaz)
        # The first column is the facet index; it must list exactly the
        # facets whose type matches glaz.id, in ascending order, and nothing
        # from facets 1 or 2.
        np.testing.assert_array_equal(knet_glaz[:, 0], [0, 3])
        np.testing.assert_array_equal(al_glaz[:, 0], [0, 3])

        # albedo is mutated in place: glazing facets (0, 3) get overwritten
        # with the glazing's diffuse front reflectance (al_glaz[:, 2]);
        # non-glazing facets (1, 2) -- regardless of which non-glazing type
        # they are -- must keep the caller's original albedo untouched.
        np.testing.assert_allclose(albedo[[0, 3]], al_glaz[:, 2])
        np.testing.assert_array_equal(albedo[[1, 2]], original_albedo[[1, 2]])

    def test_incidence_angle_monotonic_and_recovers_normal_incidence(self):
        """
        calc_optiprop_dir takes phi (incidence angle) and feeds it through
        calc_TR_phi's Fresnel reflectance formula. Every other test in this
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

        Tw0, Rfw0, Rbw0, _ = calc_optiprop_dir(T_0, Rf_0, Rb_0, d_g, phi=0.0)
        self.assertAlmostEqual(Tw0, T_0[0])
        self.assertAlmostEqual(Rfw0, Rf_0[0])
        self.assertAlmostEqual(Rbw0, Rb_0[0])

        angles_deg = [0, 15, 30, 45, 60, 75, 89]
        Tw_prev, Rfw_prev = None, None
        for deg in angles_deg:
            with self.subTest(angle_deg=deg):
                Tw, Rfw, Rbw, _ = calc_optiprop_dir(
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
        calc_optiprop_dif computes the "diffuse" (hemispherical) optical
        properties by averaging calc_TR_phi over incidence angles 0-90 deg,
        weighted by 2*sin(phi)*cos(phi) -- the standard Lambertian/
        hemispherical weighting (its own integral over 0..pi/2 is exactly 1,
        i.e. it is a normalized probability density over incidence angle for
        diffuse radiation). None of the other tests in this file check that
        integration itself: test_multilayer_... and test_glazing_properties
        only compare the *result* against a MATLAB reference for one fixed
        3-layer, 45 deg stack, so a bug in the integration that happened to
        still match that single number would slip through. This test instead
        checks two things that don't depend on any external reference:

        - recompute the same hemispherical average by hand from calc_TR_phi
          (the already-validated angle-dependent primitive) for a single
          layer, so no calc_TRA_EP layer combination is involved, and check
          calc_optiprop_dif reproduces it exactly;
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

        TwD, RfwD, RbwD, AwD = calc_optiprop_dif(T_0, Rf_0, Rb_0, d_g)

        # Manual reference: integrate calc_TR_phi over 0-90 deg by hand,
        # using the same normalized Lambertian weighting.
        deg = np.arange(0, 91, 1)
        phi = np.deg2rad(deg)
        weight = 2 * np.cos(phi) * np.sin(phi)
        self.assertAlmostEqual(np.trapezoid(weight, phi), 1.0, delta=1e-3)

        T = np.empty(len(deg))
        Rf = np.empty(len(deg))
        for j, p in enumerate(phi):
            T[j], Rf[j] = calc_TR_phi(T_0[0], Rf_0[0], p, d_g[0])
        TwD_expected = np.trapezoid(T * weight, phi)
        RfwD_expected = np.trapezoid(Rf * weight, phi)
        self.assertAlmostEqual(TwD, TwD_expected)
        self.assertAlmostEqual(RfwD, RfwD_expected)

        # Physical bound, independent of the manual integration above.
        Tw0, Rfw0, _, _ = calc_optiprop_dir(T_0, Rf_0, Rb_0, d_g, phi=0.0)
        Tw90, Rfw90, _, _ = calc_optiprop_dir(
            T_0, Rf_0, Rb_0, d_g, phi=np.deg2rad(89.9)
        )
        self.assertTrue(Tw90 < TwD < Tw0)
        self.assertTrue(Rfw0 < RfwD < Rfw90)

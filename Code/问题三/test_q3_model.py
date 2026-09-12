"""Meaningful correctness checks for settlement, information boundaries, and feasibility.
Run from this directory: python -m unittest -v test_q3_model.py
"""
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from q3_run_model import (Config, Data, E_MIN, E_MAX, T, dispatch, forecast_at,
                          risk_profile, settlement, solve_remaining)


def total(fees):
    return sum(fees[k] for k in ('plan_cost','up_cost','down_penalty','emergency_cost'))-fees['refund']


class Q3Tests(unittest.TestCase):
    def test_downward_settlement_and_repeated_revision(self):
        p, q0, r = np.ones(1), np.array([100.]), np.zeros(1)
        self.assertAlmostEqual(total(settlement(p,q0,np.array([80.]),r,'no_refund'))[0],110)
        self.assertAlmostEqual(total(settlement(p,q0,np.array([80.]),r,'refund'))[0],90)
        # 100 -> 150 -> 120 -> 130: final cost 100+1.5*30, not sum of deviations.
        self.assertAlmostEqual(total(settlement(p,q0,np.array([130.]),r,'no_refund'))[0],145)

    def test_downward_dominance_and_refund_optimizer(self):
        cfg = Config(terminal_penalty=0, reserve_weight=0)
        kw = dict(price=np.ones(1),load=np.zeros(1),pv=np.zeros(1),start_soc=E_MAX,
                  margin=np.zeros(1),reserve=np.zeros(1),q0=np.array([100.]))
        main = solve_remaining(**kw,cfg=cfg)
        refund = solve_remaining(**kw,cfg=replace(cfg,settlement='refund'))
        self.assertAlmostEqual(main['q'][0],100,places=4)
        self.assertAlmostEqual(refund['q'][0],0,places=4)

    def test_forecast_release_and_actual_data_causality(self):
        dates = pd.date_range('2025-01-01',periods=10)
        load, pv = np.full((10,T),3000.), np.full((10,T),600.)
        forecasts = {(d,h): np.full(24,1200.) for d in dates for h in (0,6,12,18)}
        data = Data(dates,np.ones(T),np.ones(T)*3000,load,pv,forecasts)
        cfg = Config(load_bias_weight=.5)
        before = forecast_at(data,7,6,cfg)
        # Perturb all unavailable realized outcomes and all later forecast releases.
        data.load[7,36:] = 1e6
        data.pv[7,36:] = 1e6
        data.load[8:] = 1e6
        data.pv[8:] = 1e6
        data.forecasts[(dates[7],12)] = np.full(24,1e6)
        data.forecasts[(dates[7],18)] = np.full(24,1e6)
        after = forecast_at(data,7,6,cfg)
        for a,b in zip(before,after):
            np.testing.assert_array_equal(a,b)
        # 06:10 interpolates known 06:00=600 and issued 07:00=1200 -> 700 kW.
        self.assertAlmostEqual(before[1][0]*6,700)
        self.assertAlmostEqual(before[1][5]*6,1200)
        # The 18:00 row's lead6 maps to SAME-day 24:00; lead24 to next-day18:00.
        data.forecasts[(dates[7],18)] = np.arange(1,25)*100.
        self.assertAlmostEqual(forecast_at(data,7,18,Config())[1][-1]*6,600.)

    def test_reserve_subtracts_actual_margin_before_accumulation(self):
        history = np.tile([10.,-30.,50.],(10,1))
        cfg = Config(reserve_horizon=3)
        margin,reserve = risk_profile(history,np.zeros(3),np.zeros(3),.8,cfg)
        np.testing.assert_allclose(margin,[10,0,50])
        np.testing.assert_allclose(reserve,0)

    def test_beginning_soc_reserve_not_end_soc(self):
        cfg = Config(terminal_penalty=0)
        sol = solve_remaining(np.ones(1),np.zeros(1),np.zeros(1),E_MIN,
                              np.zeros(1),np.array([90.]),cfg)
        # Charging in this interval cannot create a reserve at its beginning.
        self.assertAlmostEqual(sol['slack'][0],100,places=4)

    def test_feedback_physics_at_boundaries(self):
        cfg = Config()
        c,d,e,r,w = dispatch(0,1000,0,E_MIN,cfg)
        self.assertEqual((c,d,e,r,w),(0,0,E_MIN,1000,0))
        c,d,e,r,w = dispatch(1000,0,0,E_MAX,cfg)
        self.assertEqual((c,d,e,r,w),(0,0,E_MAX,0,1000))
        e=6000.
        rng = np.random.default_rng(7)
        for _ in range(500):
            q,l,g = rng.uniform(0,1500,3)
            c,d,end,r,w = dispatch(q,l,g,e,cfg)
            self.assertAlmostEqual(q+g+d+r-l-c-w,0,places=9)
            self.assertGreaterEqual(end,E_MIN-1e-8)
            self.assertLessEqual(end,E_MAX+1e-8)
            self.assertEqual(c*d,0)
            self.assertLessEqual(max(c,d),5000/6)
            e=end


if __name__ == '__main__':
    unittest.main()

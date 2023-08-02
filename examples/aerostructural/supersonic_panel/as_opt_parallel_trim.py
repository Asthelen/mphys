import numpy as np
import openmdao.api as om

from mphys import Multipoint, MultipointParallel
from mphys.trimmed_analysis import TrimmedAnalysis

from structures_mphys import StructBuilder
from aerodynamics_mphys import AeroBuilder
from xfer_mphys import XferBuilder
from geometry_morph import GeometryBuilder
from trim_balance import TrimBalance

use_parallel = False # True=use parallel multipoint, False=run scenarios serially
use_trimming = True # True=trim CL via aoa, False=treat aoa as DVs and CL as constraints
check_totals = True # True=check objective/constraint derivatives, False=run optimization

# panel geometry
panel_chord = 0.3
panel_width = 0.01

# panel discretization
N_el_struct = 20
N_el_aero = 7

# scenario names and operating conditions
scenario_names = ['aerostructural1', 'aerostructural2']
qdyn = [3E4,1E4]
mach = [5.,3.]
aoa = [3.,2.] # only used if use_trimming=False
target_CL = [0.15,0.45] # enforced inherently if use_trimming=True, otherwise enforced by the optimizer

# material properties
material_density = 2800.
material_yield_stress = 270E6
material_modulus = 70E9

# Mphys parallel multipoint scenarios
class AerostructParallel(MultipointParallel if use_parallel else Multipoint):
    def initialize(self):
        self.options.declare('aero_builder')
        self.options.declare('struct_builder')
        self.options.declare('xfer_builder')
        self.options.declare('geometry_builder')
        self.options.declare('scenario_names')
        self.options.declare('balance_component')
        self.options.declare('balance_inputs', default=['C_L'])
        self.options.declare('balance_outputs', default=['aoa'])

    def setup(self):
        for i in range(len(self.options['scenario_names'])):
            coupling_nonlinear_solver = om.NonlinearBlockGS(maxiter=100, iprint=2, use_aitken=True, aitken_initial_factor=0.5)
            coupling_linear_solver = om.LinearBlockGS(maxiter=40, iprint=2, use_aitken=True, aitken_initial_factor=0.5)
            trim_nonlinear_solver = om.NonlinearSchurSolver(atol=1e-8, rtol=1e-8, maxiter=10, max_sub_solves=60)
            self.add_subsystem(self.options['scenario_names'][i],
                                    TrimmedAnalysis(
                                        aero_builder=self.options['aero_builder'],
                                        struct_builder=self.options['struct_builder'],
                                        ldxfer_builder=self.options['xfer_builder'],
                                        geometry_builder=self.options['geometry_builder'],
                                        balance_component=self.options['balance_component'],
                                        balance_inputs=self.options['balance_inputs'],
                                        balance_outputs=self.options['balance_outputs'],
                                        in_MultipointParallel=True,
                                        coupling_nonlinear_solver=coupling_nonlinear_solver,
                                        coupling_linear_solver=coupling_linear_solver,
                                        trim_nonlinear_solver=trim_nonlinear_solver))

# OM group
class Model(om.Group):
    def setup(self):

        # ivc
        self.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
        self.ivc.add_output('modulus', val=material_modulus)
        self.ivc.add_output('yield_stress', val=material_yield_stress)
        self.ivc.add_output('density', val=material_density)
        self.ivc.add_output('mach', val=mach)
        self.ivc.add_output('qdyn', val=qdyn)
        self.ivc.add_output('geometry_morph_param', val=1.)

        # for trimming
        if use_trimming:
            self.ivc.add_output('target_CL', val=target_CL)
        else:
            self.ivc.add_output('aoa', val=aoa, units='deg')

        # create dv_struct, which is the thickness of each structural element
        thickness = 0.001*np.ones(N_el_struct)
        self.ivc.add_output('dv_struct', thickness)

        # structure setup and builder
        structure_setup = {'panel_chord'  : panel_chord,
                           'panel_width'  : panel_width,
                           'N_el'         : N_el_struct}
        struct_builder = StructBuilder(structure_setup)

        # aero builder
        aero_setup = {'panel_chord'  : panel_chord,
                      'panel_width'  : panel_width,
                      'N_el'         : N_el_aero}
        aero_builder = AeroBuilder(aero_setup)

        # xfer builder
        xfer_builder = XferBuilder(
            aero_builder=aero_builder,
            struct_builder=struct_builder
        )

        # geometry
        builders = {'struct': struct_builder, 'aero': aero_builder}
        geometry_builder = GeometryBuilder(builders)

        # trim balance component
        if use_trimming:
            trim_balance = TrimBalance()
        else:
            trim_balance = None

        # add parallel multipoint group
        self.add_subsystem('multipoint',AerostructParallel(
                                        aero_builder=aero_builder,
                                        struct_builder=struct_builder,
                                        xfer_builder=xfer_builder,
                                        geometry_builder=geometry_builder,
                                        balance_component=trim_balance,
                                        scenario_names=scenario_names))

        for i in range(len(scenario_names)):

            # connect scalar inputs to the scenario
            for var in ['modulus', 'yield_stress', 'density', 'dv_struct']:
                self.connect(var, 'multipoint.'+scenario_names[i]+'.analysis.'+var)

            # connect vector inputs
            for var in ['mach', 'qdyn']:
                self.connect(var, 'multipoint.'+scenario_names[i]+'.analysis.'+var, src_indices=[i])

            # connections for trimming
            if use_trimming:
                self.connect('target_CL', 'multipoint.'+scenario_names[i]+'.balance.target_CL', src_indices=[i])
            else:
                self.connect('aoa', 'multipoint.'+scenario_names[i]+'.analysis.aoa', src_indices=[i])

                # add CL constraint
                self.add_constraint(f'multipoint.{scenario_names[i]}.analysis.C_L', equals=target_CL[i],
                                    parallel_deriv_color='lift_cons' if use_parallel else None) # run C_L derivatives in parallel)

            # connect top-level geom parameter
            self.connect('geometry_morph_param', 'multipoint.'+scenario_names[i]+'.analysis.geometry.geometry_morph_param')

            # add objective
            if i==0:
                self.add_objective(f'multipoint.{scenario_names[i]}.analysis.mass', ref=0.01)

            # add stress constraint
            self.add_constraint(f'multipoint.{scenario_names[i]}.analysis.func_struct', upper=1.0,
                                parallel_deriv_color='struct_cons' if use_parallel else None) # run func_struct derivatives in parallel

        # add design vars
        self.add_design_var('geometry_morph_param', lower=0.1, upper=10.0)
        self.add_design_var('dv_struct', lower=1.e-4, upper=1.e-2, ref=1.e-3)
        if not use_trimming:
            self.add_design_var('aoa', lower=-20., upper=20.)

def get_model():
    return Model()

# run model and check derivatives
if __name__ == "__main__":

    prob = om.Problem()
    prob.model = Model()

    if check_totals:
        prob.setup(mode='rev')
        om.n2(prob, show_browser=False, outfile='n2.html')
        prob.run_model()
        om.n2(prob, show_browser=False, outfile='n2_wVals.html')
        prob.check_totals(step_calc='rel_avg',
                          compact_print=True,
                          directional=False,
                          show_progress=True)

    else:

        # setup optimization driver
        prob.driver = om.ScipyOptimizeDriver(debug_print=['nl_cons','objs','desvars','totals'])
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-5
        prob.driver.options['disp'] = True
        prob.driver.options['maxiter'] = 300

        # add optimization recorder
        prob.driver.recording_options['record_objectives'] = True
        prob.driver.recording_options['record_constraints'] = True
        prob.driver.recording_options['record_desvars'] = True
        prob.driver.recording_options['record_derivatives'] = True

        recorder = om.SqliteRecorder("optimization_history.sql")
        prob.driver.add_recorder(recorder)

        # run the optimization
        prob.setup(mode='rev')
        prob.run_driver()
        prob.cleanup()

        if prob.model.comm.rank==0: # write out data
            cr = om.CaseReader("optimization_history.sql")
            driver_cases = cr.list_cases('driver')

            case = cr.get_case(0)
            cons = case.get_constraints()
            dvs = case.get_design_vars()
            objs = case.get_objectives()

            f = open("optimization_history.dat","w+")

            for i, k in enumerate(objs.keys()):
                f.write('objective: ' + k + '\n')
                for j, case_id in enumerate(driver_cases):
                    f.write(str(j) + ' ' + str(cr.get_case(case_id).get_objectives(scaled=False)[k][0]) + '\n')
                f.write(' ' + '\n')

            for i, k in enumerate(cons.keys()):
                f.write('constraint: ' + k + '\n')
                for j, case_id in enumerate(driver_cases):
                    f.write(str(j) + ' ' + ' '.join(map(str,cr.get_case(case_id).get_constraints(scaled=False)[k])) + '\n')
                f.write(' ' + '\n')

            for i, k in enumerate(dvs.keys()):
                f.write('DV: ' + k + '\n')
                for j, case_id in enumerate(driver_cases):
                    f.write(str(j) + ' ' + ' '.join(map(str,cr.get_case(case_id).get_design_vars(scaled=False)[k])) + '\n')
                f.write(' ' + '\n')

            f.close()

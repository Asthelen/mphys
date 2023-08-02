import openmdao.api as om
from mphys import Multipoint
from mphys.scenario_aerostructural import ScenarioAeroStructural
from mphys.scenario_aerodynamic import ScenarioAerodynamic

from copy import deepcopy

class TrimmedAnalysis(Multipoint):
    def initialize(self):
        self.options.declare('aero_builder', default=None, recordable=False)
        self.options.declare('struct_builder', default=None, recordable=False)
        self.options.declare('ldxfer_builder', default=None, recordable=False)
        self.options.declare('geometry_builder', default=None, recordable=False)
        self.options.declare('controls_builder', default=None, recordable=False)
        self.options.declare('coupling_nonlinear_solver', default=None, desc='Nonlinear solver for inner aerostructural scenario')
        self.options.declare('coupling_linear_solver', default=None, desc='Linear solver for inner aerostructural scenario')
        self.options.declare('trim_nonlinear_solver', default=None, desc='Nonlinear solver for outer trim solve')
        self.options.declare('trim_linear_solver', default=None, desc='Linear solver for outer trim solve')
        self.options.declare('in_MultipointParallel', default=False)
        self.options.declare('balance_component', default=None, desc='Implicit component or group of implicit components that define trim equations to solve')
        self.options.declare('balance_inputs', default=[], desc='List of inputs to send from analysis to balance_component')
        self.options.declare('balance_outputs', default=[], desc='List of outputs to send from balance_component back to analysis')
        self.options.declare('balance_output_bounds', default=None, desc='Bounds for the trim outputs (in format {"lower": [lb1,lb2,...], "upper": [ub1,ub2,...]})')
        self.options.declare('analysis_inputs', default=None, desc='Optional list of analysis input names to connect with balance outputs, in case they differ in name')
        self.options.declare('analysis_outputs', default=None, desc='Optional list of analysis output names to connect with balance inputs, in case they differ in name')
        self.options.declare('pre_coupling_order', default=["aero", "struct", "ldxfer"])
        self.options.declare('post_coupling_order', default=["ldxfer", "aero", "struct"])
        self.options.declare('coupling_group_type', default='full_coupling')
    def setup(self):
        trim_nonlinear_solver = self.options['trim_nonlinear_solver']
        trim_linear_solver = self.options['trim_linear_solver']
        balance_component = deepcopy(self.options['balance_component'])
        balance_inputs = self.options['balance_inputs']
        balance_outputs = self.options['balance_outputs']
        balance_output_bounds = self.options['balance_output_bounds']
        analysis_inputs = self.options['analysis_inputs']
        analysis_outputs = self.options['analysis_outputs']

        if self.options['struct_builder'] is not None and self.options['aero_builder'] is not None:
            self.mphys_add_scenario('analysis',
                                    ScenarioAeroStructural(
                                        aero_builder=self.options['aero_builder'],
                                        struct_builder=self.options['struct_builder'],
                                        ldxfer_builder=self.options['ldxfer_builder'],
                                        geometry_builder=self.options['geometry_builder'],
                                        controls_builder=self.options['controls_builder'],
                                        in_MultipointParallel=self.options['in_MultipointParallel'],
                                        pre_coupling_order=self.options['pre_coupling_order'],
                                        post_coupling_order=self.options['post_coupling_order'],
                                        coupling_group_type=self.options['coupling_group_type']),
                                    coupling_nonlinear_solver=self.options['coupling_nonlinear_solver'],
                                    coupling_linear_solver=self.options['coupling_linear_solver'])

        elif self.options['aero_builder'] is not None: # aero only
            self.mphys_add_scenario('analysis',
                                    ScenarioAerodynamic(
                                        aero_builder=self.options['aero_builder'],
                                        geometry_builder=self.options['geometry_builder'],
                                        controls_builder=self.options['controls_builder'],
                                        in_MultipointParallel=self.options['in_MultipointParallel']))

        elif self.options['struct_builder'] is not None: # struct only
            raise NotImplementedError('Structures-only trimmed scenario has not been implemented')

        if balance_component is not None: # add balance component (None balance_component just for
                                          # troubleshooting/ensuring setup is okay before trying to trim)
            self.add_subsystem('balance', balance_component)

            # set solver options
            if trim_nonlinear_solver is not None:
                trim_nonlinear_solver._groupNames = ['analysis','balance']
                trim_nonlinear_solver._mode_nonlinear = 'rev'
                trim_nonlinear_solver.options['solve_subsystems'] = True
            else:
                trim_nonlinear_solver = om.NonlinearSchurSolver(
                                                atol=1e-10,
                                                rtol=1e-10,
                                                solve_subsystems=True,
                                                maxiter=60,
                                                max_sub_solves=60,
                                                mode_nonlinear='rev',
                                                groupNames=["analysis", "balance"],
                                                bounds=balance_output_bounds)

            if trim_linear_solver is not None:
                trim_linear_solver._groupNames = ['analysis','balance']
                trim_linear_solver._mode_linear = 'rev'
            else:
                trim_linear_solver = om.LinearSchur(mode_linear='rev', groupNames=["analysis", "balance"])

            # set solvers
            self.nonlinear_solver = trim_nonlinear_solver
            self.linear_solver = trim_linear_solver

            # for handling coupled analysis/balance inputs/outputs that differ by name
            analysis_inputs = self.options['analysis_inputs']
            analysis_outputs = self.options['analysis_outputs']
            if analysis_inputs is None:
                analysis_inputs = balance_outputs
            if analysis_outputs is None:
                analysis_outputs = balance_inputs

            # connect from scenario to balance component
            for analysis_var, balance_var in zip(analysis_outputs, balance_inputs):
                self.connect(f'analysis.{analysis_var}', f'balance.{balance_var}')

            # connect from balance component back to scenario
            for balance_var, analysis_var in zip(balance_outputs, analysis_inputs):
                self.connect(f'balance.{balance_var}', f'analysis.{analysis_var}')

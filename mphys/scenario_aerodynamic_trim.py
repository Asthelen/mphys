from .scenario import Scenario
import openmdao.api as om

class ScenarioAerodynamicTrim(Scenario):
    def initialize(self):
        """
        A class to perform a single discipline aerodynamic case.
        The Scenario will add the aerodynamic builder's precoupling subsystem,
        the coupling subsystem, and the postcoupling subsystem.
        """
        super().initialize()

        self.options.declare('aero_builder', recordable=False,
                             desc='The MPhys builder for the aerodynamic solver')
        self.options.declare('trim_builder',recordable=False,
                             desc='The MPhys builder for the trim balance equations')
        self.options.declare('in_MultipointParallel', default=False, types=bool,
                             desc='Set to `True` if adding this scenario inside a MultipointParallel Group.')
        self.options.declare('geometry_builder', default=None, recordable=False,
                             desc='The optional MPhys builder for the geometry')
        self.options.declare('controls_builder', default=None, recordable=False,
                             desc='The optional MPhys builder for control surfaces')
        self.options.declare('trim_nonlinear_solver', default=None,
                             desc='The optional nonlinear Schur solver')
        self.options.declare('trim_linear_solver', default=None,
                             desc='The optional linear Schur solver')
        self.options.declare('trim_bounds', default=None,
                             desc='The optional bounds for trim variables')

    def _mphys_scenario_setup(self):
        aero_builder = self.options['aero_builder']
        geometry_builder = self.options['geometry_builder']
        controls_builder = self.options['controls_builder']

        if self.options['in_MultipointParallel']:
            aero_builder.initialize(self.comm)

            if geometry_builder is not None:
                geometry_builder.initialize(self.comm)
                self.add_subsystem('mesh',aero_builder.get_mesh_coordinate_subsystem(self.name))
                self.mphys_add_subsystem('geometry',geometry_builder.get_mesh_coordinate_subsystem(self.name))
                self.connect('mesh.x_aero0','geometry.x_aero_in')
            else:
                self.mphys_add_subsystem('mesh',aero_builder.get_mesh_coordinate_subsystem(self.name))
            self.connect('x_aero0','x_aero')

        self._mphys_add_pre_coupling_subsystem_from_builder('aero', aero_builder, self.name)
        if controls_builder is not None:
            self._mphys_add_pre_coupling_subsystem_from_builder('controls', controls_builder, self.name)
        self._mphys_add_trim_group()
        self._mphys_add_post_coupling_subsystem_from_builder('aero', aero_builder, self.name)

    def _mphys_add_trim_group(self):
        controls_builder = self.options['controls_builder']
        aero_builder = self.options['aero_builder']
        trim_builder = self.options['trim_builder']
        trim_nonlinear_solver = self.options['trim_nonlinear_solver']
        trim_linear_solver = self.options['trim_linear_solver']
        trim_bounds = self.options['trim_bounds']

        # inner analysis group
        analysis_group = om.Group()
        if controls_builder is not None:
            analysis_group.add_subsystem('controls', controls_builder.get_coupling_group_subsystem(self.name), promotes=['*'])
        analysis_group.add_subsystem('aero', aero_builder.get_coupling_group_subsystem(self.name), promotes=['*'])

        # outer trim group with analysis and trim balance coupling
        coupling_group = om.Group()
        coupling_group.add_subsystem('analysis', analysis_group, promotes=['*'])
        coupling_group.add_subsystem('trim', trim_builder.get_coupling_group_subsystem(self.name), promotes=['*'])

        # set solver options
        if trim_nonlinear_solver is not None:
            trim_nonlinear_solver._groupNames = ['analysis','trim']
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
                                            groupNames=['analysis', 'trim'],
                                            bounds=trim_bounds)

        if trim_linear_solver is not None:
            trim_linear_solver._groupNames = ['analysis','trim']
            trim_linear_solver._mode_linear = 'rev'
        else:
            trim_linear_solver = om.LinearSchur(mode_linear='rev', groupNames=['analysis', 'trim'])

        coupling_group.nonlinear_solver = trim_nonlinear_solver
        coupling_group.linear_solver = trim_linear_solver
        coupling_group.set_solver_print(level=2, depth=4)

        self.mphys_add_subsystem('coupling', coupling_group)

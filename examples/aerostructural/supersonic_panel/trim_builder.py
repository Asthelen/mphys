import openmdao.api as om
from mphys import Builder

class TrimBalance(om.ImplicitComponent):
    def setup(self):
        self.add_input('C_L', tags=['mphys_coupling'])
        self.add_input('target_CL', tags=['mphys_coupling'])
        self.add_output('aoa', units='deg', tags=['mphys_coupling'])
    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="cs")
    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['aoa'] = inputs['C_L'] - inputs['target_CL']

class TrimBuilder(Builder):
    def get_coupling_group_subsystem(self, scenario_name=None):
        return TrimBalance()

import openmdao.api as om
class TrimBalance(om.ImplicitComponent):
    # case-specific trim balance component
    def setup(self):
        self.add_input('C_L')
        self.add_input('target_CL')
        self.add_output('aoa', units='deg')
    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="cs")
    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['aoa'] = inputs['C_L'] - inputs['target_CL']

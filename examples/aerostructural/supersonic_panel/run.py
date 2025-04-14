import os

import numpy as np
import openmdao.api as om
from aerodynamics_mphys import AeroBuilder
from geometry_morph import GeometryBuilder
from structures_mphys import StructBuilder
from xfer_mphys import XferBuilder

from mphys import MPhysVariables, Multipoint
from mphys.scenarios.aerostructural import ScenarioAeroStructural


class Model(Multipoint):
    def initialize(self):
        self.options.declare("scenario_name", default="aerostructural")

    def setup(self):
        # panel geometry
        panel_chord = 0.3
        panel_width = 0.01

        # panel discretization
        N_el_struct = 20
        N_el_aero = 7

        self.scenario_name = self.options["scenario_name"]

        # ivc
        self.add_subsystem("ivc", om.IndepVarComp(), promotes=["*"])
        self.ivc.add_output("modulus", val=70e9)
        self.ivc.add_output("yield_stress", val=270e6)
        self.ivc.add_output("density", val=2800.0)
        self.ivc.add_output("mach", val=5.0)
        self.ivc.add_output("qdyn", val=3e4)
        self.ivc.add_output("aoa", val=3.0, units="deg")
        self.ivc.add_output("geometry_morph_param", val=1.0)

        # create dv_struct, which is the thickness of each structural element
        thickness = 0.001 * np.ones(N_el_struct)
        self.ivc.add_output("dv_struct", thickness)

        ivc_vars = [
            "modulus",
            "yield_stress",
            "density",
            "mach",
            "qdyn",
            "aoa",
            "dv_struct",
        ]

        # structure setup and builder
        structure_setup = {
            "panel_chord": panel_chord,
            "panel_width": panel_width,
            "N_el": N_el_struct,
        }

        struct_builder = StructBuilder(structure_setup)
        struct_builder.initialize(self.comm)

        # aero builder
        aero_setup = {
            "panel_chord": panel_chord,
            "panel_width": panel_width,
            "N_el": N_el_aero,
        }

        aero_builder = AeroBuilder(aero_setup)
        aero_builder.initialize(self.comm)

        # xfer builder
        xfer_builder = XferBuilder(
            aero_builder=aero_builder, struct_builder=struct_builder
        )
        xfer_builder.initialize(self.comm)

        # geometry builder
        geometry_builder = GeometryBuilder()
        geometry_builder.add_discipline(
            struct_builder, MPhysVariables.Structures.Geometry
        )
        geometry_builder.add_discipline(
            aero_builder, MPhysVariables.Aerodynamics.Surface.Geometry
        )
        geometry_builder.initialize(self.comm)

        self.add_subsystem(
            "struct_mesh", struct_builder.get_mesh_coordinate_subsystem()
        )
        self.add_subsystem("aero_mesh", aero_builder.get_mesh_coordinate_subsystem())
        self.add_subsystem("geometry", geometry_builder.get_mesh_coordinate_subsystem())
        self.connect("geometry_morph_param", "geometry.geometry_morph_param")

        # create the run directory
        if self.comm.rank == 0:
            if not os.path.isdir(self.scenario_name):
                os.mkdir(self.scenario_name)
        self.comm.Barrier()

        # aerostructural analysis
        nonlinear_solver = om.NonlinearBlockGS(
            maxiter=100, iprint=2, use_aitken=True, aitken_initial_factor=0.5
        )
        linear_solver = om.LinearBlockGS(
            maxiter=40, iprint=2, use_aitken=True, aitken_initial_factor=0.5
        )
        self.mphys_add_scenario(
            self.scenario_name,
            ScenarioAeroStructural(
                aero_builder=aero_builder,
                struct_builder=struct_builder,
                ldxfer_builder=xfer_builder,
                run_directory=self.scenario_name,
            ),
            coupling_nonlinear_solver=nonlinear_solver,
            coupling_linear_solver=linear_solver,
        )

        for var in ivc_vars:
            self.connect(var, f"{self.scenario_name}.{var}")

        self.connect(
            f"aero_mesh.{MPhysVariables.Aerodynamics.Surface.Mesh.COORDINATES}",
            f"geometry.{MPhysVariables.Aerodynamics.Surface.Geometry.COORDINATES_INPUT}",
        )
        self.connect(
            f"struct_mesh.{MPhysVariables.Structures.Mesh.COORDINATES}",
            f"geometry.{MPhysVariables.Structures.Geometry.COORDINATES_INPUT}",
        )

        self.connect(
            f"geometry.{MPhysVariables.Aerodynamics.Surface.Geometry.COORDINATES_OUTPUT}",
            f"{self.scenario_name}.{MPhysVariables.Aerodynamics.Surface.COORDINATES_INITIAL}",
        )
        self.connect(
            f"geometry.{MPhysVariables.Structures.Geometry.COORDINATES_OUTPUT}",
            f"{self.scenario_name}.{MPhysVariables.Structures.COORDINATES}",
        )

        # add design variables, to simplify remote setup
        self.add_design_var("geometry_morph_param", lower=0.1, upper=10.0)
        self.add_design_var("dv_struct", lower=1.0e-4, upper=1.0e-2, ref=1.0e-3)


def get_model(scenario_name):
    return Model(scenario_name=scenario_name[0])


# run model and check derivatives
if __name__ == "__main__":

    prob = om.Problem()
    prob.model = Model()
    prob.setup(mode="rev")

    om.n2(prob, show_browser=False, outfile="n2.html")

    prob.run_model()
    print("mass =        " + str(prob["aerostructural.mass"]))
    print("func_struct = " + str(prob["aerostructural.func_struct"]))
    print("C_L =         " + str(prob["aerostructural.C_L"]))

    prob.check_totals(
        of=["aerostructural.mass", "aerostructural.func_struct", "aerostructural.C_L"],
        wrt=[
            "modulus",
            "yield_stress",
            "density",
            "mach",
            "qdyn",
            "aoa",
            "dv_struct",
            "geometry_morph_param",
        ],
        step_calc="rel_avg",
        compact_print=True,
    )

    prob.check_partials(compact_print=True, step_calc="rel_avg")

[Mesh]
  [FileInput]
    type = FileMeshGenerator
    file = /Users/hadrienrattez/Desktop/WinstonMS/Meshes/Active/Scan005_002.e
    #second_order = true
  []
  [TopNode]
    type = ExtraNodesetGenerator
    input = FileInput
    new_boundary = 'top_node'
    nodes = '56898'
  []
[]

#[MeshModifiers]
#  [top_node]
#    type = AddExtraNodeset
#    new_boundary = 'top_node'
#    nodes = '1280'
#  []
#[]

[GlobalParams]
  displacements = 'disp_x disp_y disp_z'
[]

[Modules]
  [TensorMechanics]
    [Master]
      [all]
        strain = SMALL
        incremental = true
        # add_variables = true
        #generate_output = 'hydrostatic_stress vonmises_stress stress_xx stress_yy stress_zz strain_xx strain_yy strain_zz'
        generate_output = 'stress_xx stress_yy stress_zz strain_xx strain_yy strain_zz'
      []
    []
  []
[]

[Variables]
  [disp_x]
    #order = SECOND
  []
  [disp_y]
    #order = SECOND
  []
  [disp_z]
    #order = SECOND
  []
[]

[AuxVariables]
  [stress-strain]
    family = MONOMIAL
    order = CONSTANT
  []
[]

[AuxKernels]
  [stress-strain]
    type = RankTwoDoubleContraction
    variable = stress-strain
    rank_two_tensor1 = stress
    rank_two_tensor2 = total_strain
    execute_on = timestep_end
  []
[]

[Materials]
  [Elasticity_tensor]
    type = ComputeIsotropicElasticityTensor
    poissons_ratio = 0.35
    youngs_modulus = 640 #4.25 #2765/65*0.1
  []
  [mc]
    type = ComputeMultiPlasticityStress
    ep_plastic_tolerance = 1E-9
    plastic_models = j2
  []
  [pl_strain_rate]
    type = ComputePlasticStrainRate
  []
[]

[Functions]
  [loading_vel_y]
    type = ParsedFunction
    #value = '-0.1/60*t/48*1.5/2'
    value = '-5/60*t/22*1.5/2'
    # -vel(mm/min)/60*t/real_sample_length*numerical_sample_length
  []
  [time_fct]
    type = PiecewiseLinear
    y = '0.0015 0.000025'
    x = '0 0.0039'
  []
[]

[BCs]
  [uy]
    type = DirichletBC
    variable = disp_y
    boundary = 'bottom'
    value = 0
  []
  [uy_load]
    type = FunctionDirichletBC
    variable = disp_y
    boundary = 'top'
    function = loading_vel_y
  []
  [uz]
    type = DirichletBC
    variable = disp_z
    boundary = 'top_node' #'bottom top'
    value = 0
  []
  [ux]
    type = DirichletBC
    variable = disp_x
    boundary = 'top_node' #'bottom top'
    value = 0
  []
[]

[Preconditioning]
  # active = ''
  [SMP]
    type = SMP
    full = true
  []
[]

[Executioner]
  type = Transient
  l_max_its = 50
  nl_max_its = 15
  solve_type = NEWTON
  petsc_options_iname = '-pc_type -pc_hypre_type -ksp_gmres_restart'
  petsc_options_value = 'hypre boomeramg 201'
  nl_abs_tol = 1e-5
  nl_rel_tol = 1e-5
  l_tol = 1e-2
  reset_dt = true
  line_search = bt
  start_time = 0.0
  end_time = 0.25 #0.02 #1
  #abort_on_solve_fail = true
  dt = 0.001 #0.0005 #0.015 #
  # [TimeStepper]
  #   type = FunctionDT
  #   function = time_fct
  # []
  # [TimeStepper]
  #   type = SolutionTimeAdaptiveDT
  #   dt = 0.005
  # []
[]

[Postprocessors]
  [volume]
    type = VolumePostprocessor
    execute_on = 'TIMESTEP_END INITIAL'
    use_displaced_mesh = true
  []
  #[area]
  #  type = AreaPostprocessor
  #  execute_on = 'TIMESTEP_END INITIAL'
  #  use_displaced_mesh = true
  #  boundary = out
  #[]
  #[stress_00]
  #  type = MaterialTensorIntegral
  #  rank_two_tensor = stress
  #  index_i = 0
  #  index_j = 0
  #[]
  [stress_11_top]
    type = SideAverageValue
    variable = stress_yy
    boundary = top
  []
  #[stress_22]
  #  type = MaterialTensorIntegral
  #  rank_two_tensor = stress
  #  index_i = 2
  #  index_j = 2
  #[]
  #[strain_00]
  #  type = MaterialTensorSideIntegral
  #  rank_two_tensor = total_strain
  #  index_i = 0
  #  index_j = 0
  #  boundary = out
  #[]
  #[strain_11]
  #  type = MaterialTensorSideIntegral
  #  rank_two_tensor = total_strain
  #  index_i = 1
  #  index_j = 1
  #  boundary = out
  #[]
  #[strain_22]
  #  type = MaterialTensorSideIntegral
  #  rank_two_tensor = total_strain
  #  index_i = 2
  #  index_j = 2
  #  boundary = out
  #[]
  #[chain_force]
  #  type = SideExtremeValue
  #  variable = vonmises_stress
  #  boundary = 'out'
  #  value_type = max
  #[]
  #[Hills_micro]
  #  type = ElementIntegralVariablePostprocessor
  #  variable = stress-strain
  #[]
[]

[Outputs]
  #exodus = true
  csv = true
  file_base = Scan005_002
  #perf_graph = true
[]

[UserObjects]
  [str]
    type = TensorMechanicsHardeningConstant
    value = 0.1
  []
  [j2]
    type = TensorMechanicsPlasticJ2
    yield_strength = str
    yield_function_tolerance = 1E-9
    internal_constraint_tolerance = 1E-9
  []
[]

#[RankTwoScalarAction]
#  [pp]
#    rank_two_tensor = 'stress'
#    scalar_type = 'VonMisesStress Hydrostatic'
#  []
#  [pp2]
#    rank_two_tensor = 'plastic_strain_rate'
#    scalar_type = 'EffectiveStrain'
#  []
#  [pp3]
#    rank_two_tensor = 'total_strain'
#    scalar_type = 'VolumetricStrain EffectiveStrain'
#    boundary = top
#    compute_on_boundary = true
#  []
#[]

#[RankTwoScalarVoidAction]
#  [pp]
#    scalar_type = 'VolumetricStrain EffectiveStrain'
#    boundary = 10
#  []
#[]

#[RankTwoContractionAction]
#  [pp2]
#    rank_two_tensor = 'stress total_strain'
#    boundary = top
#    compute_on_boundary = true
#  []
#[]

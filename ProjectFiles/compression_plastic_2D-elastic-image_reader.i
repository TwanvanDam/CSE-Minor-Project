[Mesh]
  block_name = 'edges solid'
  block_id = '1 2'
  [./mesh]
    type = GeneratedMeshGenerator
    dim = 2
    nx = 400
    ny = 400
    elem_type = QUAD8
  [../]
  [./image]
    type = ImageSubdomainGenerator
    file = /Users/mlesueur/projects/project/moose/running/mesh-perm/output_files/gray_pil.png
    threshold = 100
    input = mesh
  [../]
  [./delete]
    type = BlockDeletionGenerator
    input = image
    block = 0
  [../]
  [./box]
    type = SubdomainBoundingBoxGenerator
    bottom_left = '0.05 0.05 0'
    top_right = '0.95 0.95 0'
    block_id = 2
    input = delete
  [../]
[]

[GlobalParams]
  displacements = 'disp_x disp_y'
[]

[Modules]
  [TensorMechanics]
    [Master]
      [all]
        strain = SMALL
        incremental = true
        generate_output = 'hydrostatic_stress vonmises_stress stress_xx stress_yy strain_xx strain_yy'
      []
    []
  []
[]

[Variables]
  [disp_x]
    order = SECOND
  []
  [disp_y]
    order = SECOND
  []
[]

[Materials]
  [Elasticity_tensor]
    type = ComputeIsotropicElasticityTensor
    poissons_ratio = 0.3
    youngs_modulus = 10
  []
  # pure elasticity
  # [stress]
  #   type = ComputeFiniteStrainElasticStress
  # []
  # elasto-plasticity
  [dp]
    type = ComputeMultiPlasticityStress
    ep_plastic_tolerance = 1E-5
    plastic_models = j2
    ignore_failures = true
  []
[]

[UserObjects]
  [./j2]
    type = TensorMechanicsPlasticJ2
    yield_strength = str
    yield_function_tolerance = 1E-5
    internal_constraint_tolerance = 1E-5
  [../]
  # [./dp_grains]
  #   type = RedbackPlasticDruckerPrager
  #   mc_interpolation_scheme = Native
  #   mc_cohesion = str
  #   mc_friction_angle = mc_phi_grains
  #   mc_dilation_angle = mc_phi_grains
  #   internal_constraint_tolerance = 1E-5
  #   yield_function_tolerance = 1E-5
  # [../]
  [./str]
    type = TensorMechanicsHardeningConstant
    value = 1
  [../]
  # [./mc_phi_grains]
  #   type = TensorMechanicsHardeningConstant
  #   value = 45
  #   convert_to_radians = true
  # [../]
[]

[BCs]
  [compression_y]
    type = DirichletBC
    variable = disp_y
    boundary = 'top'
    value = -0.0001
  []
  [compression_x]
    type = DirichletBC
    variable = disp_x
    boundary = 'right'
    value = -0.0001
  []
  [zero_disp_y]
    type = DirichletBC
    variable = disp_y
    boundary = 'bottom'
    value = 0
  []
  [zero_disp_x]
    type = DirichletBC
    variable = disp_x
    boundary = 'left'
    value = 0
  []
[]

[Postprocessors]
  [./max_von_mises]
    type = ElementExtremeMaterialProperty
    block = 2
    value_type = max
    mat_prop = vonmises_stress
  [../]
[]

[RankTwoScalarAction]
  [pp]
    rank_two_tensor = 'stress'
    scalar_type = 'VonMisesStress Hydrostatic'
  []
[]

[Preconditioning]
  [SMP]
    type = SMP
    full = true
  []
[]

[Executioner]
  type = Steady
  solve_type = NEWTON
  petsc_options_iname = '-pc_type -pc_hypre_type -ksp_gmres_restart'
  petsc_options_value = 'hypre boomeramg 201'
  l_max_its = 50
  nl_max_its = 15
  nl_abs_tol = 1e-6
  nl_rel_tol = 1e-5
  l_tol = 1e-2
  line_search = bt

  # solve_type = NEWTON
  # line_search = bt
  # petsc_options_iname = '-pc_type -pc_factor_mat_solver_package '
  # petsc_options_value = '   lu       superlu_dist'
  #
  # l_tol = 1e-03
  # nl_abs_tol = 1e-6
  # nl_rel_tol = 1e-08
  # nl_max_its = 15
  # l_max_its = 20
[]

[Outputs]
  exodus = true
  file_base = 2D_iso-compression_elastic
[]

# second step of creating a mesh from CT scan image with optimal refinement
# or input file to create a mesh from CT scan image with no adaptivity
# This generates the file containing the mesh with 2blocks (and possibly the level of refinement/adaptivity that we want).

[Mesh]
  block_name = 'pores grains'
  block_id = '0 1'
  boundary_name = grains_edges
  boundary_id = 10
  [./generate]
    #type = FileMesh
    #file = image_mesh_canvas.e
    type = GeneratedMeshGenerator
    dim = 3
    nx = 50
    ny = 50
    nz = 50
    elem_type = HEX
  [../]
  [./image]
    type = ImageSubdomainGenerator
    input = generate
    file_suffix = tif
    threshold = 90
    #the image folder and the images selected
    file_base = /Users/hadrienrattez/Desktop/WinstonMS/Meshes/Stacks/Scan001_007/Scan001_007_
    #file_base = TestStack/png/teststack_
    #file_range = '32'
  [../]
  #tags the interface pore-grain
  #the interface's normal points outward from pore to grains
  #[./interface]
  #  type = SideSetsBetweenSubdomainsGenerator
  #  input = image
  #  primary_block = 0
  #  paired_block = 1
  #  new_boundary = 10
  #[../]
  #deletes one of the blocks
  [./delete]
    type = BlockDeletionGenerator
    input = image
    #depends_on = interface
    block_id = 0
  [../]
[]

[Variables]
  [./u]
  [../]
[]

[Problem]
  type = FEProblem
  solve = false
[]

[Executioner]
  type = Steady
[]

[Outputs]
  file_base = Scan001_007
  execute_on = 'timestep_end'
  exodus = true
[]

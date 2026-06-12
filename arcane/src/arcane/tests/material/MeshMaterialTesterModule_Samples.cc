// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialTesterModule_Samples.cc                         (C) 2000-2026 */
/*                                                                           */
/* Material manager test module.                                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/SimdOperation.h"

#include "arcane/core/BasicModule.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/materials/MaterialVariableBuildInfo.h"
#include "arcane/core/materials/IMeshBlock.h"
#include "arcane/core/materials/MeshMaterialVariableRef.h"
#include "arcane/core/materials/MeshEnvironmentVariableRef.h"

#include "arcane/materials/MeshBlockBuildInfo.h"
#include "arcane/materials/MeshEnvironmentBuildInfo.h"
#include "arcane/materials/MatCellVector.h"
#include "arcane/materials/EnvCellVector.h"
#include "arcane/materials/MeshMaterialModifier.h"
#include "arcane/materials/MatConcurrency.h"
#include "arcane/materials/ComponentSimd.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Example for documentation. Must compile, but will not be executed.
class Sample
: public Arcane::BasicModule
{
 public:

  explicit Sample(const Arcane::ModuleBuildInfo& mbi)
  : BasicModule(mbi)
  , m_mat_density(Arcane::MaterialVariableBuildInfo(0, "TestDensity"))
  {}
  Arcane::Materials::MaterialVariableCellReal m_mat_density;

  //![SampleDependenciesComputeFunction]
  void _computeDensity(Arcane::Materials::IMeshMaterial* mat)
  {
    ENUMERATE_MATCELL (imc, mat) {
      Arcane::MatCell mc = *imc;
      m_mat_density[mc] = 1.0;
    }

    // Indicates that the variable is up to date.
    m_mat_density.setUpToDate(mat);
  }
  //![SampleDependenciesComputeFunction]

  void _sample()
  {
    //![SampleMaterial]
    //![SampleMaterialCreate]
    // Creation or retrieval of the manager from a mesh.
    Arcane::Materials::IMeshMaterialMng* material_mng = nullptr;
    material_mng = Arcane::Materials::IMeshMaterialMng::getReference(defaultMesh());
    //![SampleMaterialCreate]

    //![SampleMaterialCreate2]
    // Example of creating 3 materials:
    material_mng->registerMaterialInfo("MAT1");
    material_mng->registerMaterialInfo("MAT2");
    material_mng->registerMaterialInfo("MAT3");
    //![SampleMaterialCreate2]

    //![SampleMaterialCreate3]
    // Creation of environment ENV1 containing materials MAT1 and MAT2
    Arcane::Materials::MeshEnvironmentBuildInfo ebi1("ENV1");
    ebi1.addMaterial("MAT1");
    ebi1.addMaterial("MAT2");
    Arcane::Materials::IMeshEnvironment* env1 = material_mng->createEnvironment(ebi1);

    // Creation of environment ENV2 containing material MAT2
    Arcane::Materials::MeshEnvironmentBuildInfo ebi2("ENV2");
    ebi2.addMaterial("MAT2");
    Arcane::Materials::IMeshEnvironment* env2 = material_mng->createEnvironment(ebi2);

    // Creation of environment ENV3 containing materials MAT3 and MAT1
    Arcane::Materials::MeshEnvironmentBuildInfo ebi3("ENV3");
    ebi3.addMaterial("MAT3");
    ebi3.addMaterial("MAT1");
    Arcane::Materials::IMeshEnvironment* env3 = material_mng->createEnvironment(ebi3);

    // Creation of block BLOCK1 on the group of all cells
    // and containing environments ENV1 and ENV2
    Arcane::Materials::MeshBlockBuildInfo mb1("BLOCK1", allCells());
    mb1.addEnvironment(env1);
    mb1.addEnvironment(env2);
    Arcane::Materials::IMeshBlock* block = material_mng->createBlock(mb1);

    // Indicates to the manager that initialization is complete
    material_mng->endCreate();
    //![SampleMaterialCreate3]

    //![SampleMaterialCreate4]
    info() << env1->id(); // Displays '0'
    info() << env1->materials()[0]->id(); // Displays '0'
    info() << env1->materials()[1]->id(); // Displays '1'
    info() << env2->id(); // Displays '1'
    info() << env2->materials()[0]->id(); // Displays '2'
    info() << env3->id(); // Displays '2'
    info() << env3->materials()[0]->id(); // Displays '3'
    info() << env3->materials()[1]->id(); // Displays '4'
    info() << block->id(); // Displays '0'
    //![SampleMaterialCreate4]

    //![SampleMaterialAddMat]
    {
      // Created the modification instance. The modifications
      // will be effective when the destructor of
      // this class is called.
      Arcane::Materials::MeshMaterialModifier modifier(material_mng);
      // Adds cells of material 1 or 2 based on
      // their localId()
      Arcane::UniqueArray<Arcane::Int32> mat1_indexes;
      Arcane::UniqueArray<Arcane::Int32> mat2_indexes;
      const Arcane::Int32 nb_cell = allCells().size();
      ENUMERATE_CELL (icell, allCells()) {
        Arcane::Int32 local_id = icell.itemLocalId();
        Arcane::Int32 z = icell.index();
        bool add_to_mat1 = (z < (nb_cell / 2) && z > (nb_cell / 4));
        bool add_to_mat2 = (z >= (nb_cell / 2) || z < (nb_cell / 3));
        if (add_to_mat1)
          mat1_indexes.add(local_id);
        if (add_to_mat2)
          mat2_indexes.add(local_id);
      }
      // Adds cells of material 1
      modifier.addCells(env1->materials()[0], mat1_indexes);
      // Adds cells of material 2
      modifier.addCells(env1->materials()[1], mat2_indexes);
    }
    // From here, the materials are updated.
    info() << env1->materials()[0]->cells().size(); // Number of cells for the material
    //![SampleMaterialAddMat]

    //![SampleMaterialCreateVariable]
    Arcane::IMesh* mesh = defaultMesh();
    Arcane::Materials::MaterialVariableBuildInfo mvbinfo(material_mng, "Density");
    Arcane::Materials::MaterialVariableCellReal mat_density(mvbinfo);
    Arcane::Materials::MaterialVariableCellReal mat_pressure(Arcane::VariableBuildInfo(mesh, "Pressure"));
    //![SampleMaterialCreateVariable]

    //![SampleMaterialIterEnv]
    // Iteration over all environments, then all materials and
    // all cells of this material
    ENUMERATE_ENV (ienv, material_mng) {
      Arcane::Materials::IMeshEnvironment* env = *ienv;
      ENUMERATE_MAT (imat, env) {
        Arcane::Materials::IMeshMaterial* mat = *imat;
        ENUMERATE_MATCELL (imatcell, mat) {
          Arcane::Materials::MatCell mc = *imatcell;
          info() << "Cell mat=" << mc.materialId();
        }
      }
      ENUMERATE_ENVCELL (ienvcell, env) {
        Arcane::Materials::EnvCell mmcell = *ienvcell;
        info() << "Cell env=" << mmcell.environmentId();
      }
    }
    //![SampleMaterialIterEnv]

    //![SampleBlockEnvironmentIter]
    // Iteration over all cells of materials in a block's environments.
    ENUMERATE_ENV (ienv, block) {
      Arcane::Materials::IMeshEnvironment* env = *ienv;
      ENUMERATE_MAT (imat, env) {
        Arcane::Materials::IMeshMaterial* mat = *imat;
        ENUMERATE_MATCELL (imatcell, mat) {
          Arcane::Materials::MatCell mc = *imatcell;
          info() << "Cell mat=" << mc.materialId();
        }
      }
    }
    //![SampleBlockEnvironmentIter]

    //![SampleMaterialIterCell]
    // Iteration over all environments and all materials of a cell.
    ENUMERATE_ALLENVCELL (iallenvcell, material_mng, allCells()) {
      Arcane::Materials::AllEnvCell all_env_cell = *iallenvcell;
      ENUMERATE_CELL_ENVCELL (ienvcell, all_env_cell) {
        Arcane::Materials::EnvCell env_cell = *ienvcell;
        info() << "Cell env=" << env_cell.environmentId();
        ENUMERATE_CELL_MATCELL (imatcell, env_cell) {
          Arcane::Materials::MatCell mc = *imatcell;
          info() << "Cell mat=" << mc.materialId();
        }
      }
    }
    //![SampleMaterialIterCell]

    //![SampleBlockMaterialIterCell]
    // Iteration over all environments and all materials of a cell.
    ENUMERATE_ALLENVCELL (iallenvcell, block) {
      Arcane::AllEnvCell all_env_cell = *iallenvcell;
      ENUMERATE_CELL_ENVCELL (ienvcell, all_env_cell) {
        Arcane::EnvCell env_cell = *ienvcell;
        info() << "Cell env=" << env_cell.environmentId();
        ENUMERATE_CELL_MATCELL (imatcell, env_cell) {
          Arcane::MatCell mc = *imatcell;
          info() << "Cell mat=" << mc.materialId();
        }
      }
    }
    //![SampleBlockMaterialIterCell]

    //![SampleMaterialIterFromGroup]
    Arcane::CellGroup cells;
    Arcane::Materials::IMeshMaterial* mat = env1->materials()[0];
    Arcane::Materials::MatCellVector mat_cells(cells, mat);
    ENUMERATE_MATCELL (imatcell, mat_cells) {
      mat_density[imatcell] = 2.3;
    }
    Arcane::Materials::IMeshEnvironment* env = env1;
    Arcane::Materials::EnvCellVector env_cells(cells, env);
    ENUMERATE_ENVCELL (imatcell, env_cells) {
      mat_density[imatcell] = 3.1;
    }
    //![SampleMaterialIterFromGroup]

    //![SampleMaterial]

    //![SampleComponentIter]
    // Iteration over all environments, then all materials and
    // all cells of this material via the ComponentCell
    ENUMERATE_ENV (ienv, material_mng) {
      Arcane::Materials::IMeshEnvironment* env = *ienv;
      ENUMERATE_MAT (imat, env) {
        Arcane::Materials::IMeshMaterial* mat = *imat;
        ENUMERATE_COMPONENTCELL (iccell, mat) {
          Arcane::Materials::ComponentCell cc = *iccell;
          info() << "Cell mat=" << cc.componentId();
          mat_density[cc] = 3.1; // Updates the material density
        }
      }
      ENUMERATE_COMPONENTCELL (iccell, env) {
        Arcane::Materials::ComponentCell cc = *iccell;
        info() << "Cell env=" << cc.componentId();
        mat_density[cc] = 2.5; // Updates the environment density
      }
    }
    //![SampleComponentIter]

    {
      //![SampleComponentSuperItem]
      Arcane::Materials::MatCell mc;
      Arcane::Materials::ComponentCell cc = mc;
      // Returns the environment cell (EnvCell) of the material:
      Arcane::Materials::ComponentCell cc2 = cc.superCell();
      // Iterates over the material cells of the environment
      ENUMERATE_CELL_COMPONENTCELL (icc, cc2) {
      }

      // Returns the AllEnvCell cell of the environment:
      Arcane::Materials::ComponentCell cc3 = cc2.superCell();
      // Iterates over the environment cells of the cell.
      ENUMERATE_CELL_COMPONENTCELL (icc, cc3) {
      }
      //![SampleComponentSuperItem]
    }

    {
      Arcane::Real init_val = 0.0;
      Arcane::Materials::MaterialVariableCellReal& var = mat_density;
      // Initializes the global value
      var.globalVariable().fill(init_val);
      ENUMERATE_ENV (ienv, material_mng) {
        // Initializes the environment values
        ENUMERATE_ENVCELL (ienvcell, (*ienv)) {
          var[ienvcell] = init_val;
        }
        // Initializes the material values
        ENUMERATE_MAT (imat, (*ienv)) {
          ENUMERATE_MATCELL (imatcell, (*imat)) {
            var[imatcell] = init_val;
          }
        }
      }
    }

    {
      //![SampleDependencies]
      // Sets the calculation method.
      mat_density.setMaterialComputeFunction(this, &Sample::_computeDensity);
      // Adds dependency on a material variable
      mat_density.addMaterialDepend(mat_pressure);
      // Adds dependency on global variables
      mat_density.addMaterialDepend(defaultMesh()->nodesCoordinates());
      mat_density.addMaterialDepend(m_global_time);

      ENUMERATE_MAT (imat, material_mng) {
        Arcane::Materials::IMeshMaterial* mat = *imat;
        // Updates the variable on the material \a mat if needed.
        mat_density.update(mat);
      }
      //![SampleDependencies]
    }

    {
      //![SampleConcurrency]
      // Parallel loop over the environment cells of env1
      Arcane::Materials::IMeshEnvironment* env = env1;
      Arcane::Parallel::Foreach(env->envView(), [&](Arcane::Materials::EnvItemVectorView view) {
        ENUMERATE_ENVCELL (ienvcell, view) {
          mat_density[ienvcell] = 2.5;
        }
      });

      // Parallel loop over the cells of the first material in env1
      Arcane::Materials::IMeshMaterial* mat = env1->materials()[0];
      Arcane::Parallel::Foreach(mat->matView(), [&](Arcane::Materials::MatItemVectorView view) {
        ENUMERATE_MATCELL (imatcell, view) {
          mat_density[imatcell] = 2.5;
        }
      });

      // Generic functor on a material or environment.
      auto func = [&](Arcane::Materials::ComponentItemVectorView view) {
        ENUMERATE_COMPONENTCELL (iccell, view) {
          mat_density[iccell] = 2.5;
        }
      };
      // Parallel application of \a func on the material
      Arcane::Parallel::Foreach(mat->view(), func);
      // Parallel application of \a func on the environment
      Arcane::Parallel::Foreach(env->view(), func);

      // Parallel application of \a func on the environment with options
      Arcane::ParallelLoopOptions options;
      Arcane::Parallel::Foreach(env->view(), options, func);
      //![SampleConcurrency]
    }

    {
      //![SampleEnumerateVariableDeclaration]
      Arcane::Materials::MaterialVariableCellReal mat_pressure(Arcane::VariableBuildInfo(mesh, "Pressure"));
      Arcane::Materials::MaterialVariableCellReal mat_volume(Arcane::VariableBuildInfo(mesh, "Volume"));
      Arcane::Materials::MaterialVariableCellReal mat_temperature(Arcane::VariableBuildInfo(mesh, "Temperature"));
      //![SampleEnumerateVariableDeclaration]

      //![SampleEnumerateSimdComponentItem]
      Arcane::Real nr = 1.0;
      // Temperature and volume are read-only
      auto in_volume = viewIn(mat_volume);
      auto in_temperature = viewIn(mat_temperature);
      // Pressure is writable
      auto out_pressure = viewOut(mat_pressure);

      ENUMERATE_COMPONENTITEM_LAMBDA(EnvPartSimdCell, scell, env)
      {
        out_pressure[scell] = nr * in_temperature[scell] / in_volume[scell];
      };
      //![SampleEnumerateSimdComponentItem]

      //![SampleEnumerateComponentItemEnv]
      Arcane::CellGroup test_env_group;
      Arcane::Materials::IMeshEnvironment* env = env1;
      Arcane::Materials::EnvCellVector env_vector(test_env_group, env);

      // Loop over the environment cells \a env
      ENUMERATE_COMPONENTITEM (Arcane::Materials::EnvCell, ienvcell, env) {
        Arcane::Materials::EnvCell c = *ienvcell;
        mat_pressure[c] = mat_temperature[ienvcell];
      }

      // Loop over the environment cells \a env_vector
      ENUMERATE_COMPONENTITEM (Arcane::Materials::EnvCell, ienvcell, env_vector) {
        Arcane::Materials::EnvCell c = *ienvcell;
        mat_pressure[c] = mat_temperature[ienvcell];
      }

      // Loop over the pure environment cells \a env
      ENUMERATE_COMPONENTITEM (Arcane::Materials::EnvPartCell, ienvcell, env->pureEnvItems()) {
        Arcane::Materials::EnvCell c = *ienvcell;
        mat_pressure[c] = mat_temperature[ienvcell];
      }

      // Loop over the pure environment cells \a env
      ENUMERATE_COMPONENTITEM (Arcane::Materials::EnvPartCell, ienvcell, env, Arcane::Materials::eMatPart::Pure) {
        Arcane::Materials::EnvCell c = *ienvcell;
        mat_pressure[c] = mat_temperature[ienvcell];
      }

      // Loop over the impure environment cells \a env
      ENUMERATE_COMPONENTITEM (Arcane::Materials::EnvPartCell, ienvcell, env->impureEnvItems()) {
        Arcane::Materials::EnvCell c = *ienvcell;
        mat_pressure[c] = mat_temperature[ienvcell];
      }

      // Loop over the impure environment cells \a env
      ENUMERATE_COMPONENTITEM (Arcane::Materials::EnvPartCell, ienvcell, env, Arcane::Materials::eMatPart::Impure) {
        Arcane::Materials::EnvCell c = *ienvcell;
        mat_pressure[c] = mat_temperature[ienvcell];
      }

      //![SampleEnumerateComponentItemEnv]

      //![SampleEnumerateComponentItemMat]
      Arcane::CellGroup test_mat_group;
      Arcane::Materials::IMeshMaterial* mat = env1->materials()[0];
      Arcane::Materials::MatCellVector mat_vector(test_mat_group, mat);

      // Loop over the material cells \a mat
      ENUMERATE_COMPONENTITEM (Arcane::Materials::MatCell, imatcell, mat) {
        Arcane::Materials::MatCell c = *imatcell;
        mat_pressure[c] = mat_temperature[imatcell];
      }

      // Loop over the material cells \a mat_vector
      ENUMERATE_COMPONENTITEM (Arcane::Materials::MatCell, imatcell, mat_vector) {
        Arcane::Materials::MatCell c = *imatcell;
        mat_pressure[c] = mat_temperature[imatcell];
      }

      // Loop over the pure material cells \a mat
      ENUMERATE_COMPONENTITEM (Arcane::Materials::MatPartCell, imatcell, mat->pureMatItems()) {
        Arcane::Materials::MatCell c = *imatcell;
        mat_pressure[c] = mat_temperature[imatcell];
      }

      // Loop over the pure material cells \a mat
      ENUMERATE_COMPONENTITEM (Arcane::MatPartCell, imatcell, mat, Arcane::Materials::eMatPart::Pure) {
        Arcane::MatCell c = *imatcell;
        mat_pressure[c] = mat_temperature[imatcell];
      }

      // Loop over the impure material cells \a mat
      ENUMERATE_COMPONENTITEM (Arcane::Materials::MatPartCell, imatcell, mat->impureMatItems()) {
        Arcane::Materials::MatCell c = *imatcell;
        mat_pressure[c] = mat_temperature[imatcell];
      }

      // Loop over the impure material cells \a mat
      ENUMERATE_COMPONENTITEM (Arcane::Materials::MatPartCell, imatcell, mat, Arcane::Materials::eMatPart::Impure) {
        Arcane::Materials::MatCell c = *imatcell;
        mat_pressure[c] = mat_temperature[imatcell];
      }
      //![SampleEnumerateComponentItemMat]

      //![SampleEnumerateComponentItemComponent]
      // Generic loop over the material cells \a mat
      ENUMERATE_COMPONENTITEM (Arcane::Materials::ComponentCell, iccell, mat) {
        Arcane::Materials::ComponentCell c = *iccell;
        mat_pressure[c] = mat_temperature[iccell];
      }

      // Generic loop over the material cells \a mat_vector
      ENUMERATE_COMPONENTITEM (Arcane::Materials::ComponentCell, iccell, mat_vector) {
        Arcane::Materials::ComponentCell c = *iccell;
        mat_pressure[c] = mat_temperature[iccell];
      }

      // Generic loop over the pure material cells \a mat
      ENUMERATE_COMPONENTITEM (Arcane::Materials::ComponentPartCell, iccell, mat->pureItems()) {
        Arcane::Materials::ComponentCell c = *iccell;
        mat_pressure[c] = mat_temperature[iccell];
      }

      // Generic loop over the pure material cells \a mat
      ENUMERATE_COMPONENTITEM (Arcane::Materials::ComponentPartCell, iccell, mat, Arcane::eMatPart::Pure) {
        Arcane::Materials::ComponentCell c = *iccell;
        mat_pressure[c] = mat_temperature[iccell];
      }

      // Generic loop over the impure material cells \a mat
      ENUMERATE_COMPONENTITEM (Arcane::Materials::ComponentPartCell, iccell, mat->impureItems()) {
        Arcane::Materials::ComponentCell c = *iccell;
        mat_pressure[c] = mat_temperature[iccell];
      }

      // Generic loop over the impure material cells \a mat
      ENUMERATE_COMPONENTITEM (Arcane::Materials::ComponentPartCell, iccell, mat, Arcane::eMatPart::Impure) {
        Arcane::Materials::ComponentCell c = *iccell;
        mat_pressure[c] = mat_temperature[iccell];
      }
      //![SampleEnumerateComponentItemComponent]
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

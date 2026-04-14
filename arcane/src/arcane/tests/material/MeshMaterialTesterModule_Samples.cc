// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialTesterModule_Samples.cc                         (C) 2000-2025 */
/*                                                                           */
/* Module de test du gestionnaire des matériaux.                             */
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

// Exemple pour la documentation. Doit compiler, mais ne sera pas exécuté.
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

    // Indique que la variable est à jour.
    m_mat_density.setUpToDate(mat);
  }
  //![SampleDependenciesComputeFunction]

  void _sample()
  {
    //![SampleMaterial]
    //![SampleMaterialCreate]
    // Création ou récupération du gestionnaire depuis un maillage.
    Arcane::Materials::IMeshMaterialMng* material_mng = nullptr;
    material_mng = Arcane::Materials::IMeshMaterialMng::getReference(defaultMesh());
    //![SampleMaterialCreate]

    //![SampleMaterialCreate2]
    // Exemple de création de 3 matériaux:
    material_mng->registerMaterialInfo("MAT1");
    material_mng->registerMaterialInfo("MAT2");
    material_mng->registerMaterialInfo("MAT3");
    //![SampleMaterialCreate2]

    //![SampleMaterialCreate3]
    // Création du milieu ENV1 contenant les matériaux MAT1 et MAT2
    Arcane::Materials::MeshEnvironmentBuildInfo ebi1("ENV1");
    ebi1.addMaterial("MAT1");
    ebi1.addMaterial("MAT2");
    Arcane::Materials::IMeshEnvironment* env1 = material_mng->createEnvironment(ebi1);

    // Création du milieu ENV2 contenant le matériau MAT2
    Arcane::Materials::MeshEnvironmentBuildInfo ebi2("ENV2");
    ebi2.addMaterial("MAT2");
    Arcane::Materials::IMeshEnvironment* env2 = material_mng->createEnvironment(ebi2);

    // Création du milieu ENV3 contenant les matériaux MAT3 et MAT1
    Arcane::Materials::MeshEnvironmentBuildInfo ebi3("ENV3");
    ebi3.addMaterial("MAT3");
    ebi3.addMaterial("MAT1");
    Arcane::Materials::IMeshEnvironment* env3 = material_mng->createEnvironment(ebi3);

    // Création du bloc BLOCK1 sur le groupe de toutes les mailles
    // et contenant les milieux ENV1 et ENV2
    Arcane::Materials::MeshBlockBuildInfo mb1("BLOCK1", allCells());
    mb1.addEnvironment(env1);
    mb1.addEnvironment(env2);
    Arcane::Materials::IMeshBlock* block = material_mng->createBlock(mb1);

    // Indique au gestionnaire que l'initialisation est terminée
    material_mng->endCreate();
    //![SampleMaterialCreate3]

    //![SampleMaterialCreate4]
    info() << env1->id(); // Affiche '0'
    info() << env1->materials()[0]->id(); // Affiche '0'
    info() << env1->materials()[1]->id(); // Affiche '1'
    info() << env2->id(); // Affiche '1'
    info() << env2->materials()[0]->id(); // Affiche '2'
    info() << env3->id(); // Affiche '2'
    info() << env3->materials()[0]->id(); // Affiche '3'
    info() << env3->materials()[1]->id(); // Affiche '4'
    info() << block->id(); // Affiche '0'
    //![SampleMaterialCreate4]

    //![SampleMaterialAddMat]
    {
      // Créé l'instance de modification. Les modifications
      // seront effectives lors de l'appel au destructeur de
      // cette classe.
      Arcane::Materials::MeshMaterialModifier modifier(material_mng);
      // Ajoute les mailles du matériau 1 ou 2 en fonction
      // de leur localId()
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
      // Ajoute les mailles du matériau 1
      modifier.addCells(env1->materials()[0], mat1_indexes);
      // Ajoute les mailles du matériau 2
      modifier.addCells(env1->materials()[1], mat2_indexes);
    }
    // A partir d'ici, les matériaux sont mis à jour.
    info() << env1->materials()[0]->cells().size(); // Nombre de mailles du matériau
    //![SampleMaterialAddMat]

    //![SampleMaterialCreateVariable]
    Arcane::IMesh* mesh = defaultMesh();
    Arcane::Materials::MaterialVariableBuildInfo mvbinfo(material_mng, "Density");
    Arcane::Materials::MaterialVariableCellReal mat_density(mvbinfo);
    Arcane::Materials::MaterialVariableCellReal mat_pressure(Arcane::VariableBuildInfo(mesh, "Pressure"));
    //![SampleMaterialCreateVariable]

    //![SampleMaterialIterEnv]
    // Itération sur tous les milieux, puis tous les matériaux et
    // toutes les mailles de ce matériau
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
    // Itération sur tous les mailles des matériaux des milieux d'un bloc.
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
    // Itération sur tous les milieux et tous les matériaux d'une maille.
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
    // Itération sur tous les milieux et tous les matériaux d'une maille.
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
    // Itération sur tous les milieux, puis tous les matériaux et
    // toutes les mailles de ce matériau via la ComponentCell
    ENUMERATE_ENV (ienv, material_mng) {
      Arcane::Materials::IMeshEnvironment* env = *ienv;
      ENUMERATE_MAT (imat, env) {
        Arcane::Materials::IMeshMaterial* mat = *imat;
        ENUMERATE_COMPONENTCELL (iccell, mat) {
          Arcane::Materials::ComponentCell cc = *iccell;
          info() << "Cell mat=" << cc.componentId();
          mat_density[cc] = 3.1; // Met à jour la densité du matériau
        }
      }
      ENUMERATE_COMPONENTCELL (iccell, env) {
        Arcane::Materials::ComponentCell cc = *iccell;
        info() << "Cell env=" << cc.componentId();
        mat_density[cc] = 2.5; // Met à jour la densité du milieu
      }
    }
    //![SampleComponentIter]

    {
      //![SampleComponentSuperItem]
      Arcane::Materials::MatCell mc;
      Arcane::Materials::ComponentCell cc = mc;
      // Retourne la maille milieu (EnvCell) du matériau:
      Arcane::Materials::ComponentCell cc2 = cc.superCell();
      // Itère sur les mailles matériaux du milieu
      ENUMERATE_CELL_COMPONENTCELL (icc, cc2) {
      }

      // Retourne la maille AllEnvCell du milieu:
      Arcane::Materials::ComponentCell cc3 = cc2.superCell();
      // Itère sur les mailles milieu de la maille.
      ENUMERATE_CELL_COMPONENTCELL (icc, cc3) {
      }
      //![SampleComponentSuperItem]
    }

    {
      Arcane::Real init_val = 0.0;
      Arcane::Materials::MaterialVariableCellReal& var = mat_density;
      // Initialise la valeur globale
      var.globalVariable().fill(init_val);
      ENUMERATE_ENV (ienv, material_mng) {
        // Initialise les valeurs milieux
        ENUMERATE_ENVCELL (ienvcell, (*ienv)) {
          var[ienvcell] = init_val;
        }
        // Initialise les valeurs matériaux
        ENUMERATE_MAT (imat, (*ienv)) {
          ENUMERATE_MATCELL (imatcell, (*imat)) {
            var[imatcell] = init_val;
          }
        }
      }
    }

    {
      //![SampleDependencies]
      // Positionne la méthode de calcul.
      mat_density.setMaterialComputeFunction(this, &Sample::_computeDensity);
      // Ajoute dépendance sur une variable matériau
      mat_density.addMaterialDepend(mat_pressure);
      // Ajoute dépendance sur variables globales
      mat_density.addMaterialDepend(defaultMesh()->nodesCoordinates());
      mat_density.addMaterialDepend(m_global_time);

      ENUMERATE_MAT (imat, material_mng) {
        Arcane::Materials::IMeshMaterial* mat = *imat;
        // Met à jour la variable sur le matériau \a mat si besoin.
        mat_density.update(mat);
      }
      //![SampleDependencies]
    }

    {
      //![SampleConcurrency]
      // Boucle parallèle sur les mailles du milieu env1
      Arcane::Materials::IMeshEnvironment* env = env1;
      Arcane::Parallel::Foreach(env->envView(), [&](Arcane::Materials::EnvItemVectorView view) {
        ENUMERATE_ENVCELL (ienvcell, view) {
          mat_density[ienvcell] = 2.5;
        }
      });

      // Boucle parallèle sur les mailles du premier matériaux de env1
      Arcane::Materials::IMeshMaterial* mat = env1->materials()[0];
      Arcane::Parallel::Foreach(mat->matView(), [&](Arcane::Materials::MatItemVectorView view) {
        ENUMERATE_MATCELL (imatcell, view) {
          mat_density[imatcell] = 2.5;
        }
      });

      // Functor générique sur un matériau ou milieu.
      auto func = [&](Arcane::Materials::ComponentItemVectorView view) {
        ENUMERATE_COMPONENTCELL (iccell, view) {
          mat_density[iccell] = 2.5;
        }
      };
      // Application en parallèle de \a func sur le matériau
      Arcane::Parallel::Foreach(mat->view(), func);
      // Application en parallèle de \a func sur le milieu
      Arcane::Parallel::Foreach(env->view(), func);

      // Application en parallèle de \a func sur le milieu avec options
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
      // Température et volume en lecture seule
      auto in_volume = viewIn(mat_volume);
      auto in_temperature = viewIn(mat_temperature);
      // Pression en écriture
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

      // Boucle sur les mailles du milieu \a env
      ENUMERATE_COMPONENTITEM (Arcane::Materials::EnvCell, ienvcell, env) {
        Arcane::Materials::EnvCell c = *ienvcell;
        mat_pressure[c] = mat_temperature[ienvcell];
      }

      // Boucle sur les mailles du milieu \a env_vector
      ENUMERATE_COMPONENTITEM (Arcane::Materials::EnvCell, ienvcell, env_vector) {
        Arcane::Materials::EnvCell c = *ienvcell;
        mat_pressure[c] = mat_temperature[ienvcell];
      }

      // Boucle sur les mailles pures du milieu \a env
      ENUMERATE_COMPONENTITEM (Arcane::Materials::EnvPartCell, ienvcell, env->pureEnvItems()) {
        Arcane::Materials::EnvCell c = *ienvcell;
        mat_pressure[c] = mat_temperature[ienvcell];
      }

      // Boucle sur les mailles pures du milieu \a env
      ENUMERATE_COMPONENTITEM (Arcane::Materials::EnvPartCell, ienvcell, env, Arcane::Materials::eMatPart::Pure) {
        Arcane::Materials::EnvCell c = *ienvcell;
        mat_pressure[c] = mat_temperature[ienvcell];
      }

      // Boucle sur les mailles impures du milieu \a env
      ENUMERATE_COMPONENTITEM (Arcane::Materials::EnvPartCell, ienvcell, env->impureEnvItems()) {
        Arcane::Materials::EnvCell c = *ienvcell;
        mat_pressure[c] = mat_temperature[ienvcell];
      }

      // Boucle sur les mailles impures du milieu \a env
      ENUMERATE_COMPONENTITEM (Arcane::Materials::EnvPartCell, ienvcell, env, Arcane::Materials::eMatPart::Impure) {
        Arcane::Materials::EnvCell c = *ienvcell;
        mat_pressure[c] = mat_temperature[ienvcell];
      }

      //![SampleEnumerateComponentItemEnv]

      //![SampleEnumerateComponentItemMat]
      Arcane::CellGroup test_mat_group;
      Arcane::Materials::IMeshMaterial* mat = env1->materials()[0];
      Arcane::Materials::MatCellVector mat_vector(test_mat_group, mat);

      // Boucle sur les mailles du matériau \a mat
      ENUMERATE_COMPONENTITEM (Arcane::Materials::MatCell, imatcell, mat) {
        Arcane::Materials::MatCell c = *imatcell;
        mat_pressure[c] = mat_temperature[imatcell];
      }

      // Boucle sur les mailles du matériau \a mat_vector
      ENUMERATE_COMPONENTITEM (Arcane::Materials::MatCell, imatcell, mat_vector) {
        Arcane::Materials::MatCell c = *imatcell;
        mat_pressure[c] = mat_temperature[imatcell];
      }

      // Boucle sur les mailles pures du matériau \a mat
      ENUMERATE_COMPONENTITEM (Arcane::Materials::MatPartCell, imatcell, mat->pureMatItems()) {
        Arcane::Materials::MatCell c = *imatcell;
        mat_pressure[c] = mat_temperature[imatcell];
      }

      // Boucle sur les mailles pures du matériau \a mat
      ENUMERATE_COMPONENTITEM (Arcane::MatPartCell, imatcell, mat, Arcane::Materials::eMatPart::Pure) {
        Arcane::MatCell c = *imatcell;
        mat_pressure[c] = mat_temperature[imatcell];
      }

      // Boucle sur les mailles impures du matériau \a mat
      ENUMERATE_COMPONENTITEM (Arcane::Materials::MatPartCell, imatcell, mat->impureMatItems()) {
        Arcane::Materials::MatCell c = *imatcell;
        mat_pressure[c] = mat_temperature[imatcell];
      }

      // Boucle sur les mailles impures du matériau \a mat
      ENUMERATE_COMPONENTITEM (Arcane::Materials::MatPartCell, imatcell, mat, Arcane::Materials::eMatPart::Impure) {
        Arcane::Materials::MatCell c = *imatcell;
        mat_pressure[c] = mat_temperature[imatcell];
      }
      //![SampleEnumerateComponentItemMat]

      //![SampleEnumerateComponentItemComponent]
      // Boucle générique sur les mailles du matériau \a mat
      ENUMERATE_COMPONENTITEM (Arcane::Materials::ComponentCell, iccell, mat) {
        Arcane::Materials::ComponentCell c = *iccell;
        mat_pressure[c] = mat_temperature[iccell];
      }

      // Boucle générique sur les mailles du matériau \a mat_vector
      ENUMERATE_COMPONENTITEM (Arcane::Materials::ComponentCell, iccell, mat_vector) {
        Arcane::Materials::ComponentCell c = *iccell;
        mat_pressure[c] = mat_temperature[iccell];
      }

      // Boucle générique sur les mailles pures du matériau \a mat
      ENUMERATE_COMPONENTITEM (Arcane::Materials::ComponentPartCell, iccell, mat->pureItems()) {
        Arcane::Materials::ComponentCell c = *iccell;
        mat_pressure[c] = mat_temperature[iccell];
      }

      // Boucle générique sur les mailles pures du matériau \a mat
      ENUMERATE_COMPONENTITEM (Arcane::Materials::ComponentPartCell, iccell, mat, Arcane::eMatPart::Pure) {
        Arcane::Materials::ComponentCell c = *iccell;
        mat_pressure[c] = mat_temperature[iccell];
      }

      // Boucle générique sur les mailles impures du matériau \a mat
      ENUMERATE_COMPONENTITEM (Arcane::Materials::ComponentPartCell, iccell, mat->impureItems()) {
        Arcane::Materials::ComponentCell c = *iccell;
        mat_pressure[c] = mat_temperature[iccell];
      }

      // Boucle générique sur les mailles impures du matériau \a mat
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

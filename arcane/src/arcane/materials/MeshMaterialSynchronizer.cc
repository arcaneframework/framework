// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialSynchronizer.cc                                 (C) 2000-2025 */
/*                                                                           */
/* Synchronisation des entités des matériaux.                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/internal/MeshMaterialSynchronizer.h"

#include "arcane/materials/CellToAllEnvCellConverter.h"
#include "arcane/materials/MatItemEnumerator.h"
#include "arcane/materials/MeshMaterialModifier.h"

#include "arcane/utils/HashSuite.h"

#include "arcane/core/VariableTypes.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/ItemGenericInfoListView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialSynchronizer::
MeshMaterialSynchronizer(IMeshMaterialMng* material_mng)
: TraceAccessor(material_mng->traceMng())
, m_material_mng(material_mng)
{
  if (Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_ACC_MAT_SYNCHRONIZER", true)) {
    m_synchronizer = new AcceleratorMeshMaterialSynchronizerImpl(material_mng);
    info() << "using ACC material synchronizer";
  }
  else {
    m_synchronizer = new LegacyMeshMaterialSynchronizerImpl(material_mng);
    info() << "using DEFAULT material synchronizer";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialSynchronizer::
~MeshMaterialSynchronizer()
{
  delete m_synchronizer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MeshMaterialSynchronizer::
synchronizeMaterialsInCells()
{
  return m_synchronizer->synchronizeMaterialsInCells();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie que les mailles des matériaux sont bien cohérentes entre les
 * sous-domaines.
 * Cette méthode est collective
 */
void MeshMaterialSynchronizer::
checkMaterialsInCells(Integer max_print)
{
  /*
    Pour cela, on utilise une variable aux mailles et on applique
    l'algorithme suivant pour chaque matériau:
    - le sous-domaine propriétaire remplit cette variable
    avec l'indice du matériau
    - la variable est synchronisée.
    - chaque sous-domaine vérifie ensuite pour chaque maille
    que si la variable a pour valeur l'indice du matériau, alors
    ce matériau est présent.
  */

  IMesh* mesh = m_material_mng->mesh();
  if (!mesh->parallelMng()->isParallel())
    return;
  m_material_mng->checkValid();

  info(4) << "CheckMaterialsInCells";

  VariableCellInt32 indexes(VariableBuildInfo(mesh,"ArcaneMaterialPresenceIndexes"));
  _checkComponents(indexes, m_material_mng->materialsAsComponents(), max_print);
  _checkComponents(indexes, m_material_mng->environmentsAsComponents(), max_print);

  _checkComponentsInGhostCells(indexes, max_print);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSynchronizer::
_checkComponents(VariableCellInt32& indexes,
                 ConstArrayView<IMeshComponent*> components,
                 Integer max_print)
{
  IMesh* mesh = m_material_mng->mesh();
  Integer nb_component = components.size();
  Integer nb_error = 0;

  info() << "Checking components nb=" << nb_component;

  CellToAllEnvCellConverter all_env_cell_converter(m_material_mng);

  for( Integer i=0; i<nb_component; ++i ){
    indexes.fill(-1);
    IMeshComponent* c = components[i];
    ENUMERATE_COMPONENTCELL(iccell,c){
      ComponentCell cc = *iccell;
      indexes[cc.globalCell()] = i;
    }

    indexes.synchronize();

    ENUMERATE_ALLENVCELL(iallenvcell,m_material_mng,mesh->allCells()){
      AllEnvCell all_env_cell = *iallenvcell;
      Cell cell = all_env_cell.globalCell();
      bool has_sync_mat = (indexes[cell]==i);
      ComponentCell cc = c->findComponentCell(all_env_cell);
      bool has_component = !cc.null();
      if (has_sync_mat!=has_component){
        ++nb_error;
        if (max_print<0 || nb_error<max_print)
          error() << "Bad component synchronisation for i=" << i
                  << " name=" << c->name()
                  << " cell_uid=" << cell.uniqueId()
                  << " sync_mat=" << has_sync_mat
                  << " has_component=" << has_component;
      }
    }
  }
  if (nb_error!=0)
    ARCANE_FATAL("Bad synchronisation");
}

void MeshMaterialSynchronizer::
_checkComponentsInGhostCells(VariableCellInt32& hashes, Integer max_print)
{
  IMesh* mesh = m_material_mng->mesh();
  Integer nb_error = 0;

  ENUMERATE_ALLENVCELL (iallenvcell, m_material_mng, mesh->ownCells()) {
    AllEnvCell all_env_cell = *iallenvcell;
    Cell cell = all_env_cell.globalCell();

    IntegerHashSuiteT<Int32> hash_suite;

    Int32 nb_env = all_env_cell.nbEnvironment();
    hash_suite.add(nb_env);

    for (Integer i = 0; i < nb_env; ++i) {
      EnvCell env_cell = all_env_cell.cell(i);
      Int32 nb_matt = env_cell.nbMaterial();
      hash_suite.add(nb_matt);

      for (Integer j = 0; j < nb_matt; ++j) {
        MatCell mat = env_cell.cell(j);
        Int32 id = mat.materialId();
        hash_suite.add(id);
      }
    }

    hashes[cell] = hash_suite.hash();
  }

  hashes.synchronize();

  ENUMERATE_ALLENVCELL (iallenvcell, m_material_mng, mesh->allCells()) {
    AllEnvCell all_env_cell = *iallenvcell;
    Cell cell = all_env_cell.globalCell();
    if (cell.isOwn())
      continue;

    IntegerHashSuiteT<Int32> hash_suite;

    Int32 nb_env = all_env_cell.nbEnvironment();
    hash_suite.add(nb_env);

    for (Integer i = 0; i < nb_env; ++i) {
      EnvCell env_cell = all_env_cell.cell(i);
      Int32 nb_matt = env_cell.nbMaterial();
      hash_suite.add(nb_matt);

      for (Integer j = 0; j < nb_matt; ++j) {
        MatCell mat = env_cell.cell(j);
        Int32 id = mat.materialId();
        hash_suite.add(id);
      }
    }
    if (hashes[cell] != hash_suite.hash()) {
      nb_error++;
      if (max_print < 0 || nb_error < max_print) {
        error() << "Bad components synchronization -- Cell : " << cell << " -- Hash : " << hash_suite.hash();
      }
    }
  }
  if (nb_error != 0)
    ARCANE_FATAL("Bad components synchronization -- Nb error : {0}", nb_error);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

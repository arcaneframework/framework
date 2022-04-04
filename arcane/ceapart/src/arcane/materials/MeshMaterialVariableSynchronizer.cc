﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableSynchronizer.cc                         (C) 2000-2016 */
/*                                                                           */
/* Synchroniseur de variables matériaux.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IMesh.h"
#include "arcane/IVariableSynchronizer.h"
#include "arcane/ItemGroup.h"
#include "arcane/IParallelMng.h"
#include "arcane/IItemFamily.h"

#include "arcane/materials/MeshMaterialVariableSynchronizer.h"
#include "arcane/materials/IMeshMaterialMng.h"
#include "arcane/materials/MatItemEnumerator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialVariableSynchronizer::
MeshMaterialVariableSynchronizer(IMeshMaterialMng* material_mng,
                                 IVariableSynchronizer* var_syncer,
                                 MatVarSpace space)
: TraceAccessor(material_mng->traceMng())
, m_material_mng(material_mng)
, m_variable_synchronizer(var_syncer)
, m_timestamp(-1)
, m_var_space(space)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialVariableSynchronizer::
~MeshMaterialVariableSynchronizer()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IVariableSynchronizer* MeshMaterialVariableSynchronizer::
variableSynchronizer()
{
  return m_variable_synchronizer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<MatVarIndex> MeshMaterialVariableSynchronizer::
sharedItems(Int32 index)
{
  return m_shared_items[index];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<MatVarIndex> MeshMaterialVariableSynchronizer::
ghostItems(Int32 index)
{
  return m_ghost_items[index];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplit \a items avec la liste de tous les MatVarIndex des
 * mailles de \a view.
 */
void MeshMaterialVariableSynchronizer::
_fillCells(Array<MatVarIndex>& items,AllEnvCellVectorView view)
{
  bool has_mat = m_var_space==MatVarSpace::MaterialAndEnvironment;
  // NOTE: il est possible d'optimiser en regardant les milieux qui n'ont
  // qu'un seul matériau car dans ce cas la valeur milieu et la valeur
  // matériau est la même. De la même manière, s'il n'y a qu'un milieu
  // alors la valeur globale et milieu est la même. Dans ces cas, il n'est
  // pas nécessaire d'ajouter la deuxième MatCell dans la liste.
  items.clear();
  ENUMERATE_ALLENVCELL(iallenvcell,view){
    AllEnvCell all_env_cell = *iallenvcell;
    ENUMERATE_CELL_ENVCELL(ienvcell,all_env_cell){
      EnvCell env_cell = *ienvcell;
      items.add(ienvcell._varIndex());
      if (has_mat){
        ENUMERATE_CELL_MATCELL(imatcell,env_cell){
          items.add(imatcell._varIndex());
        }
      }
    }
    // A priori ajouter cette information n'est pas nécessaire car il
    // est possible de récupérer l'info de la variable globale.
    items.add(MatVarIndex(0,view.localId(iallenvcell.index())));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableSynchronizer::
checkRecompute()
{
  Int64 ts = m_material_mng->timestamp();
  if (m_timestamp!=ts)
    recompute();
  m_timestamp = ts;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableSynchronizer::
recompute()
{
  IVariableSynchronizer* var_syncer = m_variable_synchronizer;

  // Calcul des informations de synchronisation pour les mailles matériaux.
  // NOTE: Cette version nécessite que les matériaux soient correctement
  // synchronisés entre les sous-domaines.

  IItemFamily* family = var_syncer->itemGroup().itemFamily();
  IParallelMng* pm = var_syncer->parallelMng();
  if (!pm->isParallel())
    return;
  ItemGroup all_items = family->allItems();

  Int32ConstArrayView ranks = var_syncer->communicatingRanks();
  Integer nb_rank = ranks.size();

  m_shared_items.resize(nb_rank);
  m_ghost_items.resize(nb_rank);

  for( Integer i=0; i<nb_rank; ++i ){

    {
      Int32ConstArrayView shared_ids = var_syncer->sharedItems(i);
      CellVectorView shared_cells(family->view(shared_ids));
      AllEnvCellVectorView view = m_material_mng->view(shared_cells);
      Array<MatVarIndex>& items = m_shared_items[i];
      _fillCells(items,view);
      info(4) << "SIZE SHARED FOR rank=" << ranks[i] << " n=" << items.size();
    }

    {
      Int32ConstArrayView ghost_ids = var_syncer->ghostItems(i);
      CellVectorView ghost_cells(family->view(ghost_ids));
      AllEnvCellVectorView view = m_material_mng->view(ghost_cells);
      Array<MatVarIndex>& items = m_ghost_items[i];
      _fillCells(items,view);
      info(4) << "SIZE GHOST FOR rank=" << ranks[i] << " n=" << items.size();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshBlock.cc                                                (C) 2000-2016 */
/*                                                                           */
/* Bloc d'un maillage.                                                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArgumentException.h"

#include "arcane/IMesh.h"

#include "arcane/materials/MeshBlock.h"
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshBlock::
MeshBlock(IMeshMaterialMng* mm,Int32 block_id,const MeshBlockBuildInfo& info)
: TraceAccessor(mm->traceMng())
, m_material_mng(mm)
, m_block_id(block_id)
, m_name(info.name())
, m_cells(info.cells())
, m_environments(info.environments())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshBlock::
build()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AllEnvCellVectorView MeshBlock::
view()
{
  return m_material_mng->view(m_cells);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajoute le milieu \a env au bloc.
 *
 * Cela ne peut se faire que lors de la phase d'initialisation
 * (avant que IMeshMaterialMng::endCreate() ait été appelé).
 */
void MeshBlock::
addEnvironment(IMeshEnvironment* env)
{
  if (m_environments.contains(env))
    throw ArgumentException(A_FUNCINFO,
                            String::format("environment {0} already in block {1}",
                                           env->name(),this->name()));
  m_environments.add(env);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Supprime le milieu \a env au bloc.
 *
 * Cela ne peut se faire que lors de la phase d'initialisation
 * (avant que IMeshMaterialMng::endCreate() ait été appelé).
 */
void MeshBlock::
removeEnvironment(IMeshEnvironment* env)
{
  Integer index = -1;
  for( Integer i=0, n=m_environments.size(); i<n; ++i )
    if (m_environments[i]==env){
      index = i;
      break;
    }
  if (index==(-1))
    throw ArgumentException(A_FUNCINFO,
                            String::format("environment {0} not in block {1}",
                                           env->name(),this->name()));
  m_environments.remove(index);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

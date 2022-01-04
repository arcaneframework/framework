// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshCompactMng.cc                                           (C) 2000-2020 */
/*                                                                           */
/* Gestionnaire des compactages de familles d'un maillage.                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/MeshCompactMng.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/TraceInfo.h"

#include "arcane/IParallelMng.h"

#include "arcane/mesh/MeshCompacter.h"
#include "arcane/mesh/DynamicMesh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshCompactMng::
MeshCompactMng(IMesh* mesh)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_compacter(nullptr)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshCompactMng::
~MeshCompactMng()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshCompacter* MeshCompactMng::
beginCompact()
{
  if (m_compacter)
    ARCANE_FATAL("Already compacting");
  MeshCompacter* c = new MeshCompacter(m_mesh,m_mesh->parallelMng()->timeStats());
  return _setCompacter(c);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshCompacter* MeshCompactMng::
beginCompact(IItemFamily* family)
{
  if (m_compacter)
    ARCANE_FATAL("Already compacting");
  MeshCompacter* c = new MeshCompacter(family,m_mesh->parallelMng()->timeStats());
  return _setCompacter(c);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCompactMng::
endCompact()
{
  if (!m_compacter)
    ARCANE_FATAL("Can not call endCompact() without calling beginCompact() before");
  delete m_compacter;
  m_compacter = nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMesh* MeshCompactMng::
mesh() const
{
  return m_mesh;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshCompacter* MeshCompactMng::
_setCompacter(MeshCompacter* c)
{
  m_compacter = c;
  c->build();
  return c;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

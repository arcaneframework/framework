// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshExchangeMng.cc                                          (C) 2000-2024 */
/*                                                                           */
/* Gestionnaire des échanges de maillages entre sous-domaines.               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/TraceInfo.h"

#include "arcane/core/ISubDomain.h"

#include "arcane/mesh/MeshExchangeMng.h"
#include "arcane/mesh/MeshExchanger.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshExchangeMng::
MeshExchangeMng(IMesh* mesh)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_exchanger(nullptr)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshExchangeMng::
~MeshExchangeMng()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshExchanger* MeshExchangeMng::
beginExchange()
{
  if (m_exchanger)
    ARCANE_FATAL("Already in an exchange");
  IMeshExchanger* ex = _createExchanger();
  m_exchanger = ex;
  return m_exchanger;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshExchanger* MeshExchangeMng::
_createExchanger()
{
  MeshExchanger* ex = new MeshExchanger(m_mesh,m_mesh->subDomain()->timeStats());
  ex->build();
  return ex;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchangeMng::
endExchange()
{
  if (!m_exchanger)
    ARCANE_FATAL("Can not call endExchange() without calling beginExchange() before");
  delete m_exchanger;
  m_exchanger = nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IPrimaryMesh* MeshExchangeMng::
mesh() const
{
  return m_mesh->toPrimaryMesh();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshExchangeMng.h                                           (C) 2000-2024 */
/*                                                                           */
/* Gestionnaire des échanges de maillages entre sous-domaines.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESHEXCHANGEMNG_H
#define ARCANE_MESHEXCHANGEMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/core/IMeshExchangeMng.h"
#include "arcane/core/IMesh.h"
#include "arcane/mesh/MeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du gestionnaire des échanges de maillages entre
 * sous-domaines.
 *
 */
class ARCANE_MESH_EXPORT MeshExchangeMng
: public TraceAccessor
, public IMeshExchangeMng
{
 public:

  MeshExchangeMng(IMesh* mesh);
  ~MeshExchangeMng();

 public:

  IPrimaryMesh* mesh() const override;
  IMeshExchanger* beginExchange() override;
  void endExchange() override;
  IMeshExchanger* exchanger() override { return m_exchanger; }

 protected:

  virtual IMeshExchanger* _createExchanger();

 private:

  IMesh* m_mesh;
  IMeshExchanger* m_exchanger;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

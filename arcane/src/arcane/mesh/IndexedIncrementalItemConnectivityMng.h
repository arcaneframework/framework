// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IndexedIncrementalItemConnectivityMng.h                     (C) 2000-2022 */
/*                                                                           */
/* Gestionnaire de 'IIndexedIncrementalItemConnectivity'.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_INDEXEDINCREMENTALITEMCONNECTIVITYMNG_H
#define ARCANE_MESH_INDEXEDINCREMENTALITEMCONNECTIVITYMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Ref.h"
#include "arcane/IIndexedIncrementalItemConnectivityMng.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du gestionnaire des connectivités incrémentales indexées sur les entités.
 */
class ARCANE_MESH_EXPORT IndexedIncrementalItemConnectivityMng
: public TraceAccessor
, public IIndexedIncrementalItemConnectivityMng
{
 public:

  IndexedIncrementalItemConnectivityMng(ITraceMng* tm);

 public:

  Ref<IIndexedIncrementalItemConnectivity>
  findOrCreateConnectivity(IItemFamily* source, IItemFamily* target, const String& name) override;
  Ref<IIndexedIncrementalItemConnectivity> findConnectivity(const String& name) override;

 private:

  using ConnectivityMapType = std::map<String, Ref<IIndexedIncrementalItemConnectivity>>;

  ConnectivityMapType m_connectivity_map;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

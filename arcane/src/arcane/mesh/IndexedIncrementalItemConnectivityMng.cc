// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IndexedIncrementalItemConnectivityMng.cc                    (C) 2000-2024 */
/*                                                                           */
/* Gestionnaire de 'IIndexedIncrementalItemConnectivity'.                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/IndexedIncrementalItemConnectivityMng.h"

#include "arcane/core/IndexedItemConnectivityView.h"
#include "arcane/core/IIndexedIncrementalItemConnectivity.h"
#include "arcane/mesh/IncrementalItemConnectivity.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IndexedIncrementalItemConnectivity
: public IIndexedIncrementalItemConnectivity
{
 public:
  explicit IndexedIncrementalItemConnectivity(IncrementalItemConnectivity* x)
  : m_true_connectivity(x){}
 public:
  IIncrementalItemConnectivity* connectivity() override
  {
    return m_true_connectivity;
  }
  IndexedItemConnectivityViewBase view() const override
  {
    return m_true_connectivity->connectivityView();
  }
 public:
  IncrementalItemConnectivity* m_true_connectivity;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IndexedIncrementalItemConnectivityMng::
IndexedIncrementalItemConnectivityMng(ITraceMng* tm)
: TraceAccessor(tm)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IIndexedIncrementalItemConnectivity> IndexedIncrementalItemConnectivityMng::
findOrCreateConnectivity(IItemFamily* source, IItemFamily* target, const String& name)
{
  ARCANE_CHECK_POINTER(source);
  ARCANE_CHECK_POINTER(target);
  Ref<IIndexedIncrementalItemConnectivity> connectivity;
  auto x = m_connectivity_map.find(name);
  if (x != m_connectivity_map.end()) {
    connectivity = x->second;
    IIncrementalItemConnectivity* c2 = connectivity->connectivity();
    IItemFamily* old_source = c2->sourceFamily();
    IItemFamily* old_target = c2->targetFamily();
    if (old_source != source)
      ARCANE_FATAL("A connectivity with the same name '{0}' already exists but with a different source"
                   " old_source={1} new_source={2}",
                   name, old_source->name(), source->name());
    if (old_target != target)
      ARCANE_FATAL("A connectivity with the same name '{0}' already exists but with a different target"
                   " old_target={1} new_target={2}",
                   name, old_target->name(), target->name());
  }
  else {
    // Les connectivités créées sont désallouées automatiquement par les familles
    auto* true_connectivity = new mesh::IncrementalItemConnectivity(source, target, name);
    connectivity = makeRef<IIndexedIncrementalItemConnectivity>(new IndexedIncrementalItemConnectivity(true_connectivity));
    m_connectivity_map.insert(std::make_pair(name, connectivity));

    // Ajoute les entités existantes dans la connectivité.
    true_connectivity->_internalNotifySourceItemsAdded(source->allItems().view().localIds());
  }
  return connectivity;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IIndexedIncrementalItemConnectivity> IndexedIncrementalItemConnectivityMng::
findConnectivity(const String& name)
{
  auto x = m_connectivity_map.find(name);
  if (x != m_connectivity_map.end())
    return x->second;
  ARCANE_FATAL("No connectivity with name '{0}'", name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

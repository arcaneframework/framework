// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NullVariableSynchronizer.cc                                 (C) 2000-2023 */
/*                                                                           */
/* Synchronisation des variables en séquentiel.                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Event.h"

#include "arcane/IVariableSynchronizer.h"
#include "arcane/VariableSynchronizerEventArgs.h"
#include "arcane/ItemGroup.h"
#include "arcane/VariableCollection.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Synchronisation des variables en séquentiel.
 *
 * Implémente IVariableSynchronizer en séquentiel.
 *
 * Cette classe ne fait aucune opération.
 */
class NullVariableSynchronizer
: public IVariableSynchronizer
{
 public:

  NullVariableSynchronizer(IParallelMng* pm, const ItemGroup& group)
  : m_parallel_mng(pm)
  , m_item_group(group)
  {
  }

 public:

  IParallelMng* parallelMng() override
  {
    return m_parallel_mng;
  }

  const ItemGroup& itemGroup() override
  {
    return m_item_group;
  }
  void compute() override {}
  void changeLocalIds(Int32ConstArrayView old_to_new_ids) override
  {
    ARCANE_UNUSED(old_to_new_ids);
  }
  void synchronize(IVariable* var) override
  {
    ARCANE_UNUSED(var);
    if (m_on_synchronized.hasObservers()) {
      VariableSynchronizerEventArgs args(var, this, 0.0);
      m_on_synchronized.notify(args);
    }
  }
  void synchronize(VariableCollection vars) override
  {
    ARCANE_UNUSED(vars);
    if (m_on_synchronized.hasObservers()) {
      VariableSynchronizerEventArgs args(vars, this, 0.0);
      m_on_synchronized.notify(args);
    }
  }
  Int32ConstArrayView communicatingRanks() override
  {
    return Int32ConstArrayView();
  }

  Int32ConstArrayView sharedItems(Int32 index) override
  {
    ARCANE_UNUSED(index);
    return Int32ConstArrayView();
  }

  Int32ConstArrayView ghostItems(Int32 index) override
  {
    ARCANE_UNUSED(index);
    return Int32ConstArrayView();
  }
  void synchronizeData(IData* data) override
  {
    ARCANE_UNUSED(data);
  }

  EventObservable<const VariableSynchronizerEventArgs&>& onSynchronized() override
  {
    return m_on_synchronized;
  }

 private:

  IParallelMng* m_parallel_mng;
  ItemGroup m_item_group;
  EventObservable<const VariableSynchronizerEventArgs&> m_on_synchronized;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" IVariableSynchronizer*
createNullVariableSynchronizer(IParallelMng* pm, const ItemGroup& group)
{
  return new NullVariableSynchronizer(pm, group);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

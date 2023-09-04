// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableSynchronizerMng.h                                   (C) 2000-2023 */
/*                                                                           */
/* Gestionnaire des synchroniseurs de variables.                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_INTERNAL_VARIABLESYNCHRONIZERMNG_H
#define ARCANE_IMPL_INTERNAL_VARIABLESYNCHRONIZERMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Event.h"

#include "arcane/core/IVariableSynchronizerMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire des synchroniseurs de variables.
 */
class ARCANE_IMPL_EXPORT VariableSynchronizerMng
: public TraceAccessor
, public IVariableSynchronizerMng
{
 public:

  explicit VariableSynchronizerMng(IVariableMng* vm);

 public:

  EventObservable<const VariableSynchronizerEventArgs&>& onSynchronized() override
  {
    return m_on_synchronized;
  }

 private:

  IVariableMng* m_variable_mng = nullptr;
  EventObservable<const VariableSynchronizerEventArgs&> m_on_synchronized;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IVariableSynchronizerMng.h                                  (C) 2000-2023 */
/*                                                                           */
/* Interface du gestionnaire de synchronisation des variables.               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IVARIABLESYNCHRONIZERMNG_H
#define ARCANE_CORE_IVARIABLESYNCHRONIZERMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du gestionnaire de synchronisation des variables.
 */
class ARCANE_CORE_EXPORT IVariableSynchronizerMng
{
 public:

  virtual ~IVariableSynchronizerMng() = default;

 public:

  virtual EventObservable<const VariableSynchronizerEventArgs&>& onSynchronized() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

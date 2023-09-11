// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IVariableSynchronizerMngInternal.h                          (C) 2000-2023 */
/*                                                                           */
/* API interne à Arcane de IVariableSynchronizerMng.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_IVARIABLESYNCHRONIZERMNGINTERNAL_H
#define ARCANE_CORE_INTERNAL_IVARIABLESYNCHRONIZERMNGINTERNAL_H
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
 * \brief API interne à Arcane de IVariableSynchronizerMng.
 */
class ARCANE_CORE_EXPORT IVariableSynchronizerMngInternal
{
 public:

  virtual ~IVariableSynchronizerMngInternal() = default;

 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

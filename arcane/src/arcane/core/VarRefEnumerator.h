// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VarRefEnumerator.h                                          (C) 2000-2025 */
/*                                                                           */
/* Classe parcourant les références d'une Variable.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_VARREFENUMERATOR_H
#define ARCANE_CORE_VARREFENUMERATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IVariable.h"
#include "arcane/core/VariableRef.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VarRefEnumerator
{
 public:

  VarRefEnumerator(const IVariable* vp)
  : m_vref(vp->firstReference())
  {
  }
  void operator++()
  {
    m_vref = m_vref->nextReference();
  }
  VariableRef* operator*()
  {
    return m_vref;
  }
  bool hasNext() const
  {
    return m_vref;
  }

 private:

  VariableRef* m_vref = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

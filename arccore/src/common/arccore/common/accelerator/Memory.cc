// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Memory.cc                                                   (C) 2000-2025 */
/*                                                                           */
/* Classes de gestion mémoire associées aux accélérateurs.                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/Memory.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  const char* _toName(eMemoryAdvice r)
  {
    switch (r) {
    case eMemoryAdvice::None:
      return "None";
    case eMemoryAdvice::MostlyRead:
      return "MostlyRead";
    case eMemoryAdvice::PreferredLocationDevice:
      return "PreferredLocationDevice";
    case eMemoryAdvice::PreferredLocationHost:
      return "PreferredLocationHost";
    case eMemoryAdvice::AccessedByDevice:
      return "AccessedByDevice";
    case eMemoryAdvice::AccessedByHost:
      return "AccessedByHost";
    }
    return "Invalid";
  }
} // namespace

extern "C++" ARCCORE_COMMON_EXPORT std::ostream&
operator<<(std::ostream& o, eMemoryAdvice a)
{
  o << _toName(a);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

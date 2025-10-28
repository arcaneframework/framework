// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CommonGlobal.cc                                             (C) 2000-2025 */
/*                                                                           */
/* Définitions globales de la composante 'Common' de 'Arccore'.              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/CommonGlobal.h"

#include "arccore/common/internal/IMemoryResourceMngInternal.h"
#include "arccore/common/IMemoryResourceMng.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

namespace
{
  const char* _toName(eMemoryResource r)
  {
    switch (r) {
    case eMemoryResource::Unknown:
      return "Unknown";
    case eMemoryResource::Host:
      return "Host";
    case eMemoryResource::HostPinned:
      return "HostPinned";
    case eMemoryResource::Device:
      return "Device";
    case eMemoryResource::UnifiedMemory:
      return "UnifiedMemory";
    }
    return "Invalid";
  }

} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_COMMON_EXPORT std::ostream&
operator<<(std::ostream& o, eMemoryResource v)
{
  o << _toName(v);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

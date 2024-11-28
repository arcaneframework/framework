// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CollectionsGlobal.cc                                        (C) 2000-2024 */
/*                                                                           */
/* Définitions globales de la composante 'Collections' de 'Arccore'.         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/collections/CollectionsGlobal.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  const char* _toName(eHostDeviceMemoryLocation v)
  {
    switch (v) {
    case eHostDeviceMemoryLocation::Unknown:
      return "Unknown";
    case eHostDeviceMemoryLocation::Device:
      return "Device";
    case eHostDeviceMemoryLocation::Host:
      return "Host";
    case eHostDeviceMemoryLocation::ManagedMemoryDevice:
      return "ManagedMemoryDevice";
    case eHostDeviceMemoryLocation::ManagedMemoryHost:
      return "ManagedMemoryHost";
    }
    return "Invalid";
  }
} // namespace

extern "C++" ARCCORE_COLLECTIONS_EXPORT std::ostream&
operator<<(std::ostream& o, eHostDeviceMemoryLocation v)
{
  o << _toName(v);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

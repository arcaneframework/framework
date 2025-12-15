// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Collection.cc                                               (C) 2000-2025 */
/*                                                                           */
/* Classe de base d'une collection.                                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/Collection.h"

#include "arccore/base/ArgumentException.h"
#include "arccore/base/TraceInfo.h"

// Ces fichiers ne sont pas directement utilisés ici mais permettent
// d'exporter les symboles.
#include "arccore/common/List.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

namespace
{
  void _doNoReferenceError(const void* ptr)
  {
    std::cerr << "** FATAL: Null reference.\n";
    std::cerr << "** FATAL: Trying to use an item not referenced.\n";
    std::cerr << "** FATAL: Item is located at memory address " << ptr << ".\n";
    arccoreDebugPause("arcaneNoReferenceError");
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ObjectImpl::
_noReferenceErrorCallTerminate(const void* ptr)
{
  _doNoReferenceError(ptr);
  std::terminate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C" ARCCORE_COMMON_EXPORT void
throwOutOfRangeException()
{
  std::cerr << "** FATAL: Invalid access on a collection (array, list, ...).\n";
  throw ArgumentException(A_FUNCINFO, "Bad index");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

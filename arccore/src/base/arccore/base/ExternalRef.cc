// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExternalRef.cc                                              (C) 2000-2025 */
/*                                                                           */
/* Gestion d'une référence sur un objet externe au C++.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ExternalRef.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Internal
{
namespace
{
ExternalRef::DestroyFuncType m_destroy_functor = nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ExternalRef::Handle::
~Handle()
{
  if (!handle)
    return;
  //std::cerr << "TRY DESTROY EXTERNAL Object f=" << m_destroy_functor << " h=" << handle << "\n";
  if (m_destroy_functor && handle)
    (*m_destroy_functor)(handle);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C" ARCCORE_BASE_EXPORT void
_SetExternalRefDestroyFunctor(ExternalRef::DestroyFuncType d)
{
  //std::cerr << "SET DESTROY FUNCTOR d=" << d << "\n";
  m_destroy_functor = d;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::Internal

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


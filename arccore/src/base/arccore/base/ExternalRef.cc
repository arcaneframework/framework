// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2020 IFPEN-CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExternalRef.cc                                              (C) 2000-2019 */
/*                                                                           */
/* Gestion d'une référence sur un objet externe au C++.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ExternalRef.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::Internal
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


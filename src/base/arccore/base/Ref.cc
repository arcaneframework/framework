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
/* Ref.cc                                                      (C) 2000-2019 */
/*                                                                           */
/* Gestion des références sur une instance.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/Ref.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file arccore/base/Ref.h
 *
 * \brief Gestion des références à une classe C++
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool RefBase::DeleterBase::
_destroyHandleTrue(const void* instance,Internal::ExternalRef& handle)
{
  ARCCORE_UNUSED(instance);
  //std::cerr << "DELETE SERVICE i=" << instance << " h=" << handle << "\n";
  if (handle.isValid())
    return true;
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool RefBase::DeleterBase::
_destroyHandle(const void* instance,Internal::ExternalRef& handle)
{
  return _destroyHandleTrue(instance,handle);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool RefBase::DeleterBase::
_destroyHandle(void* instance,Internal::ExternalRef& handle)
{
  return _destroyHandleTrue(instance,handle);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


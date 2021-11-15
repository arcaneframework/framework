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
/* ArrayViewCommon.h                                           (C) 2000-2021 */
/*                                                                           */
/* Déclarations communes aux classes ArrayView, ConstArrayView et Span.      */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_ARRAYVIEWCOMMON_H
#define ARCCORE_BASE_ARRAYVIEWCOMMON_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayIterator.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::impl
{

//! Sous-vue correspondant à l'interval \a index sur \a nb_interval
template<typename ViewType>
auto subViewInterval(ViewType view,
                     typename ViewType::size_type index,
                     typename ViewType::size_type nb_interval) -> ViewType
{
  using size_type = typename ViewType::size_type;
  size_type n = view.size();
  size_type isize = n / nb_interval;
  size_type ibegin = index * isize;
  // Pour le dernier interval, prend les elements restants
  if ((index+1)==nb_interval)
    isize = n - ibegin;
  return ViewType::create(view.data()+ibegin,isize);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

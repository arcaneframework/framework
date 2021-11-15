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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
/*!
 * \brief Affiche les valeurs de la vue.
 *
 * Affiche sur le flot \a o les valeurs de \a val.
 * Si \a max_print est supérieur à 0, indique le nombre maximum de valeurs
 * à afficher.
 */
template<typename ViewType> inline void
dumpArray(std::ostream& o,ViewType val,int max_print)
{
  using size_type = typename ViewType::size_type;
  size_type n = val.size();
  if (max_print>0 && n>max_print){
    // N'affiche que les (max_print/2) premiers et les (max_print/2) derniers
    // sinon si le tableau est très grand cela peut générer des
    // sorties listings énormes.
    size_type z = (max_print/2);
    size_type z2 = n - z;
    o << "[0]=\"" << val[0] << '"';
    for( size_type i=1; i<z; ++i )
      o << " [" << i << "]=\"" << val[i] << '"';
    o << " ... ... (skipping indexes " << z << " to " << z2 << " ) ... ... ";
    for( size_type i=(z2+1); i<n; ++i )
      o << " [" << i << "]=\"" << val[i] << '"';
  }
  else{
    for( size_type i=0; i<n; ++i ){
      if (i!=0)
        o << ' ';
      o << "[" << i << "]=\"" << val[i] << '"';
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Indique si les deux vues sont égales
template<typename ViewType> inline bool
areEqual(ViewType rhs, ViewType lhs)
{
  using size_type = typename ViewType::size_type;
  if (rhs.size()!=lhs.size())
    return false;
  size_type s = rhs.size();
  for( size_type i=0; i<s; ++i ){
    if (rhs[i]!=lhs[i])
      return false;
  }
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Algorithm.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Algorithmes de la STL.                                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ALGORITHM_H
#define ARCANE_CORE_ALGORITHM_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Container, class Element> inline
typename Container::const_iterator
container_find(const Container& c, const Element& elem)
{
  typedef typename Container::const_iterator const_iterator;
  const_iterator i = ARCANE_STD::find(c.begin(), c.end(), elem);
  return i;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

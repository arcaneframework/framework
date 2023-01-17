// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimdItem.cc                                                 (C) 2000-2022 */
/*                                                                           */
/* Types des entités et des énumérateurs des entités pour la vectorisation.  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"
#include "arcane/SimdItem.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Pour tester la validité de toutes les méthodes templates

template class SimdItemT<Node>;
template class SimdItemEnumeratorT<Node>;
template class SimdItemIndexT<Node>;
template class SimdItemDirectT<Node>;
template class SimdItemDirectIndexT<Node>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

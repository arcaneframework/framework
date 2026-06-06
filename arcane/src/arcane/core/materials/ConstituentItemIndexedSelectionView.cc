// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConstituentItemIndexedSelectionView.cc                      (C) 2000-2026 */
/*                                                                           */
/* View over a subset of a ConstituentItem container.                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/ConstituentItemIndexedSelectionView.h"

#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/materials/internal/IMeshMaterialMngInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstituentItemIndexedSelectionViewBase::
ConstituentItemIndexedSelectionViewBase(IndexArrayView indices)
: m_selection_view(indices)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstituentItemIndexedSelectionViewBase::
ConstituentItemIndexedSelectionViewBase(IMeshComponent* constituent, Int32 selection_size)
{
  SmallSpan<const Int32> v = constituent->materialMng()->_internalApi()->identitySelectionView();
  // Checks that the returned view has a size at least equal to the selection.
  if (v.size() < selection_size)
    ARCANE_FATAL("Invalid size for identity selection (selection_size={0} identity={1})",
                 selection_size, v.size());
  m_selection_view = SmallSpan<const Int32>{ v.data(), selection_size };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemFunctor.cc                                              (C) 2000-2025 */
/*                                                                           */
/* Functor over entities.                                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemFunctor.h"
#include "arcane/utils/Math.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AbstractItemRangeFunctor::
AbstractItemRangeFunctor(ItemVectorView items_view, Integer grain_size)
: m_items(items_view)
, m_block_size(SIMD_PADDING_SIZE)
, m_nb_block(items_view.size())
, m_block_grain_size(grain_size)
{
  // NOTE: if the range functor is used for vectorization, it must
  // that items_view.localIds() be aligned. The problem is that we do not know
  // exactly what the required alignment is. We could base it on
  // \a m_block_size and say that the alignment is m_block_size * sizeof(Int32).
  // In any case, the potential alignment problem will be detected by
  // SimdItemEnumerator.
  Integer nb_item = m_items.size();
  m_nb_block = nb_item / m_block_size;
  if ((nb_item % m_block_size) != 0)
    ++m_nb_block;

  m_block_grain_size = grain_size / m_block_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemVectorView AbstractItemRangeFunctor::
_view(Integer begin_block, Integer nb_block, Int32* true_begin) const
{
  // Convert (begin_block, nb_block) to (begin, size) corresponding to m_items.
  Integer begin = begin_block * m_block_size;
  Integer nb_item = m_items.size();
  Integer size = math::min(nb_block * m_block_size, nb_item - begin);
  if (true_begin)
    *true_begin = begin;
  return m_items.subView(begin, size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

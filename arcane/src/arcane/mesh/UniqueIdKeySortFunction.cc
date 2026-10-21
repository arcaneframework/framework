// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* UniqueIdKeySortFunction.cc                                  (C) 2000-2026 */
/*                                                                           */
/* Sort of the items of a family by uniqueId, without comparisons.           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/internal/UniqueIdKeySortFunction.h"

#include "arcane/utils/Array.h"

#include "arcane/core/ItemInternal.h"

#include <utility>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

UniqueIdKeySortFunction::
UniqueIdKeySortFunction(const String& name, std::unique_ptr<IItemInternalSortFunction> fallback)
: m_name(name)
, m_fallback(std::move(fallback))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UniqueIdKeySortFunction::
sortItems(ItemInternalMutableArrayView items)
{
  const Int32 n = items.size();
  if (n < 2) {
    m_fallback->sortItems(items);
    return;
  }

  // One pass over the items: the keys, the range they span, whether they are
  // already in order, and where the suppressed ones are.
  UniqueArray<Int64> uids(n);
  UniqueArray<Int32> pos(n);
  UniqueArray<Int32> suppressed_pos;
  Int32 nb_alive = 0;
  Int64 min_uid = 0;
  Int64 max_uid = 0;
  Int64 prev_uid = 0;
  bool is_sorted = true;
  for (Int32 i = 0; i < n; ++i) {
    const ItemInternal* item = items[i];
    if (item->isSuppressed()) {
      suppressed_pos.add(i);
      continue;
    }
    // A suppressed item before a live one means the array is not in order.
    if (!suppressed_pos.empty())
      is_sorted = false;
    const Int64 uid = item->uniqueId().asInt64();
    if (nb_alive == 0) {
      min_uid = uid;
      max_uid = uid;
    }
    else {
      if (uid < prev_uid)
        is_sorted = false;
      if (uid < min_uid)
        min_uid = uid;
      if (uid > max_uid)
        max_uid = uid;
    }
    prev_uid = uid;
    uids[nb_alive] = uid;
    pos[nb_alive] = i;
    ++nb_alive;
  }

  if (is_sorted)
    return;

  const Int64 width = (nb_alive == 0) ? 0 : (max_uid - min_uid + 1);
  // A span that does not fit an Int64 is not a range to walk digit by digit.
  if (width <= 0) {
    m_fallback->sortItems(items);
    return;
  }
  // A dense range is a permutation of itself: the sorted position of a key is
  // the key. This is checked rather than assumed: 'width == nb_alive' with
  // unique keys is what makes the range a permutation of itself.
  const bool is_dense = (nb_alive > 0 && width == static_cast<Int64>(nb_alive));

  UniqueArray<Int32> sorted_pos(nb_alive);
  if (is_dense) {
    sorted_pos.fill(-1);
    for (Int32 i = 0; i < nb_alive; ++i)
      sorted_pos[uids[i] - min_uid] = pos[i];
    // The scatter assumed the keys were distinct, which is the family's own
    // invariant. Checking it costs one pass and turns a broken invariant into
    // a slow sort instead of a wild pointer.
    for (Int32 i = 0; i < nb_alive; ++i)
      if (sorted_pos[i] < 0) {
        m_fallback->sortItems(items);
        return;
      }
  }
  else {
    Int32 nb_pass = 0;
    for (Int64 w = width - 1; w > 0; w >>= RADIX_BITS)
      ++nb_pass;
    if (nb_pass > MAX_RADIX_PASS) {
      m_fallback->sortItems(items);
      return;
    }
    _radixSort(uids.subView(0, nb_alive), pos.subView(0, nb_alive),
               min_uid, nb_pass, sorted_pos);
  }

  // One linear permutation of the pointers, live items first.
  UniqueArray<ItemInternal*> old_items(n);
  for (Int32 i = 0; i < n; ++i)
    old_items[i] = items[i];
  for (Int32 i = 0; i < nb_alive; ++i)
    items[i] = old_items[sorted_pos[i]];
  for (Int32 i = 0, ns = suppressed_pos.size(); i < ns; ++i)
    items[nb_alive + i] = old_items[suppressed_pos[i]];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UniqueIdKeySortFunction::
_radixSort(Int64ConstArrayView uids, Int32ConstArrayView pos,
           Int64 base, Int32 nb_pass, Int32ArrayView sorted_pos)
{
  const Int32 n = uids.size();
  UniqueArray<Int64> key_a(n);
  UniqueArray<Int32> val_a(n);
  UniqueArray<Int64> key_b(n);
  UniqueArray<Int32> val_b(n);
  for (Int32 i = 0; i < n; ++i) {
    key_a[i] = uids[i] - base;
    val_a[i] = pos[i];
  }
  UniqueArray<Int32> count(RADIX_SIZE);
  Int64ArrayView src_key = key_a.view();
  Int32ArrayView src_val = val_a.view();
  Int64ArrayView dst_key = key_b.view();
  Int32ArrayView dst_val = val_b.view();
  for (Int32 p = 0; p < nb_pass; ++p) {
    const Int32 shift = p * RADIX_BITS;
    count.fill(0);
    for (Int32 i = 0; i < n; ++i)
      ++count[static_cast<Int32>((src_key[i] >> shift) & (RADIX_SIZE - 1))];
    Int32 sum = 0;
    for (Int32 d = 0; d < RADIX_SIZE; ++d) {
      const Int32 c = count[d];
      count[d] = sum;
      sum += c;
    }
    for (Int32 i = 0; i < n; ++i) {
      const Int32 d = static_cast<Int32>((src_key[i] >> shift) & (RADIX_SIZE - 1));
      const Int32 k = count[d]++;
      dst_key[k] = src_key[i];
      dst_val[k] = src_val[i];
    }
    std::swap(src_key, dst_key);
    std::swap(src_val, dst_val);
  }
  for (Int32 i = 0; i < n; ++i)
    sorted_pos[i] = src_val[i];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

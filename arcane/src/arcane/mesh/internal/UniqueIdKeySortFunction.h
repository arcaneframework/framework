// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* UniqueIdKeySortFunction.h                                   (C) 2000-2026 */
/*                                                                           */
/* Sort of the items of a family by uniqueId, without comparisons.           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_INTERNAL_UNIQUEIDKEYSORTFUNCTION_H
#define ARCANE_MESH_INTERNAL_UNIQUEIDKEYSORTFUNCTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"

#include "arcane/core/ItemTypes.h"
#include "arcane/core/IItemInternalSortFunction.h"

#include <memory>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Sorts by uniqueId without comparing, when the uniqueIds allow it.
 *
 * Same order out as sorting the ItemInternal* with
 * CompareUniqueIdWithSuppression: suppressed items last, the others by
 * increasing uniqueId. What changes is how that order is reached.
 *
 * A uniqueId is an integer, so the order it defines can be read digit by digit
 * rather than through n.log(n) comparisons: two linear passes per digit, and no
 * comparison at all. The number of digits is decided at run time from the range
 * the family actually uses, so a family whose uniqueIds span a million values
 * pays two passes and not eight.
 *
 * Three properties are tested, in this order, and each one is a fact about the
 * data rather than an assumption about the mesh:
 *
 * 1. Already in order. One linear pass, and nothing is permuted at all. An
 *    initial allocation hands out uniqueIds in creation order for at least one
 *    family, so this is not a rare case.
 * 2. A dense range. If the n uniqueIds cover exactly n consecutive values then
 *    they are a permutation of that range, and the sorted position of each one
 *    is its own value: a single scatter, no digits.
 * 3. Otherwise, the radix passes over the range's width.
 *
 * A range too wide to be worth digits falls back to the comparison sort.
 *
 * Ties: uniqueIds are unique, so the only items that compare equal are
 * suppressed ones, and the caller reads only the first nbItem() entries, which
 * hold none of them. The comparison sort was not stable either, so neither
 * promised anything there.
 */
class UniqueIdKeySortFunction
: public IItemInternalSortFunction
{
  //! Digit width of a radix pass, and the counter array that goes with it
  static constexpr Int32 RADIX_BITS = 11;
  static constexpr Int32 RADIX_SIZE = 1 << RADIX_BITS;
  //! Beyond this many passes the comparison sort is the better answer
  static constexpr Int32 MAX_RADIX_PASS = 3;

 public:

  //! \a fallback is the comparison sort, used when the uniqueIds do not allow digits.
  UniqueIdKeySortFunction(const String& name, std::unique_ptr<IItemInternalSortFunction> fallback);

  const String& name() const override { return m_name; }
  void sortItems(ItemInternalMutableArrayView items) override;

 private:

  /*!
   * \brief Least-significant-digit radix sort of \a uids, carrying \a pos.
   *
   * \a base is subtracted from every key so that the passes cover the range the
   * family uses and not the magnitude of its uniqueIds.
   */
  void _radixSort(Int64ConstArrayView uids, Int32ConstArrayView pos,
                  Int64 base, Int32 nb_pass, Int32ArrayView sorted_pos);

 private:

  String m_name;
  std::unique_ptr<IItemInternalSortFunction> m_fallback;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CommonItemGroupFilterer.h                                   (C) 2000-2021 */
/*                                                                           */
/* Filtering of common groups across all parts of a mesh.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_COMMONITEMGROUPFILTERINFO_H
#define ARCANE_COMMONITEMGROUPFILTERINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"

#include "arcane/ItemGroup.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Utility class for common groups across all parts
 * of a mesh.
 *
 * To use this class, you must add the groups you wish to filter via addGroupToFilter().
 * You must then call the applyFiltering() method to perform the filtering.
 */
class ARCANE_MESH_EXPORT CommonItemGroupFilterer
{
 public:

  explicit CommonItemGroupFilterer(IItemFamily* family);

 public:

  CommonItemGroupFilterer(const CommonItemGroupFilterer& rhs) = delete;
  CommonItemGroupFilterer& operator=(const CommonItemGroupFilterer& rhs) = delete;

 public:

  void addGroupToFilter(const ItemGroup& group);
  /*!
   * \brief Filters the common groups.
   *
   * If there are groups that are not common to all ranks, an
   * exception is raised.
   */
  void applyFiltering();

  //! List of common groups sorted alphabetically.
  ItemGroupCollection sortedCommonGroups() { return m_sorted_common_groups; }

 private:

  IItemFamily* m_family;
  List<ItemGroup> m_input_groups;
  List<ItemGroup> m_sorted_common_groups;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConstituentItemVectorBuildInfo.h                            (C) 2000-2025 */
/*                                                                           */
/* Construction options for 'ComponentItemVector'.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_CONSTITUENTITEMVECTORBUILDINFO_H
#define ARCANE_CORE_MATERIALS_CONSTITUENTITEMVECTORBUILDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemGroup.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Construction options for 'ComponentItemVector' and
 *
 * This class allows specifying the arguments to construct
 * instances of ComponentItemVector, MatCellVector or EnvCellVector.
 *
 * It is possible to construct it from a CellGroup, a CellVectorView
 * or a list of localIds().
 */
class ARCANE_CORE_EXPORT ConstituentItemVectorBuildInfo
{
  friend class MatCellVector;
  friend class EnvCellVector;

 private:

  //! Type used to construct the instance.
  enum class eBuildListType
  {
    Group,
    VectorView,
    LocalIds
  };

 public:

  //! Construction from entities of the group \a group
  explicit ConstituentItemVectorBuildInfo(const CellGroup& group);
  //! Construction from entities of the view \a view
  explicit ConstituentItemVectorBuildInfo(CellVectorView view);
  //! Construction from entities having local numbers \a local_ids
  explicit ConstituentItemVectorBuildInfo(SmallSpan<const Int32> local_ids);

 private:

  SmallSpan<const Int32> _localIds() const;

 private:

  CellGroup m_group;
  CellVectorView m_view;
  SmallSpan<const Int32> m_local_ids;
  eBuildListType m_build_list_type;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

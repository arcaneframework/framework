// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConstituentItemVectorBuildInfo.cc                           (C) 2000-2025 */
/*                                                                           */
/* Options de construction pour 'ComponentItemVector'.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/ConstituentItemVectorBuildInfo.h"

#include "arcane/utils/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstituentItemVectorBuildInfo::
ConstituentItemVectorBuildInfo(const CellGroup& group)
: m_group(group)
, m_build_list_type(eBuildListType::Group)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstituentItemVectorBuildInfo::
ConstituentItemVectorBuildInfo(CellVectorView view)
: m_view(view)
, m_build_list_type(eBuildListType::VectorView)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstituentItemVectorBuildInfo::
ConstituentItemVectorBuildInfo(SmallSpan<const Int32> local_ids)
: m_local_ids(local_ids)
, m_build_list_type(eBuildListType::LocalIds)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SmallSpan<const Int32> ConstituentItemVectorBuildInfo::
_localIds() const
{
  switch (m_build_list_type) {
  case eBuildListType::Group:
    return m_group.view().localIds();
  case eBuildListType::VectorView:
    return m_view.localIds();
  case eBuildListType::LocalIds:
    return m_local_ids;
  }
  ARCANE_FATAL("Bad value '{0}' for build type", static_cast<int>(m_build_list_type));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

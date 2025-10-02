// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConstituentItemVectorBuildInfo.h                            (C) 2000-2025 */
/*                                                                           */
/* Options de construction pour 'ComponentItemVector'.                       */
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
 * \brief Options de construction pour 'ComponentItemVector' et l
 *
 * Cette classe permet de spécifier les arguments pour construire les
 * instances de ComponentItemVector, MatCellVector ou EnvCellVector.
 *
 * Il est possible de construite à partir d'un CellGroup, d'un CellVectorView
 * ou d'une liste de localIds().
 */
class ARCANE_CORE_EXPORT ConstituentItemVectorBuildInfo
{
  friend class MatCellVector;
  friend class EnvCellVector;

 private:

  //! Type utilisé pour construite l'instance.
  enum class eBuildListType
  {
    Group,
    VectorView,
    LocalIds
  };

 public:

  //! Construction à partir des entités du groupe \a group
  explicit ConstituentItemVectorBuildInfo(const CellGroup& group);
  //! Construction à partir des entités de la vue \a view
  explicit ConstituentItemVectorBuildInfo(CellVectorView view);
  //! Construction à partir des entités ayant pour numéros locaux \a local_ids
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

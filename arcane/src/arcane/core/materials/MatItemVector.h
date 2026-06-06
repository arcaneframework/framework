// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MatItemVector.h                                             (C) 2000-2025 */
/*                                                                           */
/* Vector over the entities of a material.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_MATITEMVECTOR_H
#define ARCANE_CORE_MATERIALS_MATITEMVECTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/ComponentItemVector.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Vector over the entities of a material.
 *
 * \warning This vector is only valid as long as the material does not change.
 */
class ARCANE_CORE_EXPORT MatCellVector
: public ComponentItemVector
{
 public:

  typedef MatCellEnumerator EnumeratorType;

 public:

  //! Constructs a vector containing the entities of \a group for the material \a material.
  MatCellVector(const CellGroup& group, IMeshMaterial* material);
  //! Constructs a vector containing the entities of \a view for the material \a material.
  MatCellVector(CellVectorView view, IMeshMaterial* material);
  //! Constructs a vector containing the entities \a local_ids for the material \a material.
  MatCellVector(SmallSpan<const Int32> local_ids, IMeshMaterial* material);
  //! Constructs a vector over the entities of the material \a material.
  MatCellVector(const ConstituentItemVectorBuildInfo& build_info, IMeshMaterial* material);
  //! Copy constructor. The instance will reference \a rhs
  MatCellVector(const MatCellVector& rhs) = default;
  //! Copy constructor. This instance is a copy of \a rhs.
  MatCellVector(MatItemVectorView rhs)
  : ComponentItemVector(rhs)
  {}

 public:

  //! Conversion to a view of this vector
  operator MatCellVectorView() const
  {
    return view();
  }

  //! View of this vector
  MatCellVectorView view() const
  {
    return { _component(), _matvarIndexes(), _constituentItemListView(), _localIds() };
  }

  //! Associated material
  IMeshMaterial* material() const;

  //! Clones this vector
  MatCellVector clone() const { return { view() }; }

 private:

  void _build(SmallSpan<const Int32> view);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

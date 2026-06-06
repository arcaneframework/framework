// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* EnvItemVector.h                                             (C) 2000-2025 */
/*                                                                           */
/* Vector over the entities of an environment.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_ENVITEMVECTOR_H
#define ARCANE_CORE_MATERIALS_ENVITEMVECTOR_H
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
 * \brief Vector over the entities of an environment.
 *
 * \warning This vector is only valid as long as the environment does not change.
 */
class ARCANE_CORE_EXPORT EnvCellVector
: public ComponentItemVector
{
 public:

  //! Constructs a vector containing the entities of \a group for the environment \a environment
  EnvCellVector(const CellGroup& group, IMeshEnvironment* environment);
  //! Constructs a vector containing the entities of \a view for the environment \a environment
  EnvCellVector(CellVectorView view, IMeshEnvironment* environment);
  //! Constructs a vector containing the entities \a local_ids for the environment \a environment
  EnvCellVector(SmallSpan<const Int32> local_ids, IMeshEnvironment* environment);
  //! Constructs a vector over the entities of the environment \a environment.
  EnvCellVector(const ConstituentItemVectorBuildInfo& build_info, IMeshEnvironment* environment);
  //! Copy constructor. The instance will reference \a rhs
  EnvCellVector(const EnvCellVector& rhs) = default;
  //! Copy constructor. This instance is a copy of \a rhs.
  EnvCellVector(EnvItemVectorView rhs)
  : ComponentItemVector(rhs)
  {}

 public:

  //! Conversion to a view of this vector
  operator EnvCellVectorView() const
  {
    return view();
  }

  //! View of this vector
  EnvCellVectorView view() const
  {
    return { _component(), _matvarIndexes(), _constituentItemListView(), _localIds() };
  }

  //! Associated environment
  IMeshEnvironment* environment() const;

  //! Clone this vector
  EnvCellVector clone() const { return { view() }; }

 private:

  void _build(SmallSpan<const Int32> view);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

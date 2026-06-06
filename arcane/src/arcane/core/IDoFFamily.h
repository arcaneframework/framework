// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDoFFamily.h                                                (C) 2000-2025 */
/*                                                                           */
/* Interface of a family of degrees of freedom (DoF).                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IDOFFAMILY_H
#define ARCANE_CORE_IDOFFAMILY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Mesh
 * \brief Interface of a DoF family.
 */
class ARCANE_CORE_EXPORT IDoFFamily
{
 public:

  virtual ~IDoFFamily() = default; //<! Frees resources

 public:

  virtual void build() = 0;

 public:

  //! Name of the family
  virtual String name() const = 0;

  //! Full name of the family (including the mesh name)
  virtual String fullName() const = 0;

  //! Number of entities
  virtual Integer nbItem() const = 0;

  //! Group of all DoFs
  virtual ItemGroup allItems() const = 0;

 public:

  //! Input is the DoF uids and we retrieve their lids
  virtual DoFVectorView addDoFs(Int64ConstArrayView dof_uids, Int32ArrayView dof_lids) = 0;

  //! Adding ghosts must be followed by a call to computeSynchronizeInfos
  virtual DoFVectorView addGhostDoFs(Int64ConstArrayView dof_uids, Int32ArrayView dof_lids,
                                     Int32ConstArrayView owners) = 0;

  virtual void removeDoFs(Int32ConstArrayView items_local_id) = 0;

  /*!
   * \sa IItemFamily::endUpdate().
   */
  virtual void endUpdate() = 0;

  virtual IItemFamily* itemFamily() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

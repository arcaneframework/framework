// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemFamilyModifier.h                                       (C) 2000-2025 */
/*                                                                           */
/* Interface for modifying a family                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IITEMFAMILYMODIFIER_H
#define ARCANE_CORE_IITEMFAMILYMODIFIER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/ItemTypes.h"

#include "arcane/mesh/MeshInfos.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemTypeInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Mesh
 * \brief Interface for modifying a family.
 *
 * This class allows for the generic modification of items within a family. It
 * is used in the family modification workflow based on the dependency graph
 * (IItemFamilyNetwork).
 * It is only implemented in mesh item families (i.e., everything except
 * ParticleFamily and DoFFamily, where it is not necessary)
 */
class ARCANE_CORE_EXPORT IItemFamilyModifier
{
 public:

  /** Class destructor */
  virtual ~IItemFamilyModifier() {}

 public:

  // DEPRECATED
  ARCANE_DEPRECATED_REASON("Y2022: Use allocOne() overload with ItemTypeId")
  ItemInternal* allocOne(Int64 uid,ItemTypeInfo* type, mesh::MeshInfos& mesh_info);
  // DEPRECATED
  ARCANE_DEPRECATED_REASON("Y2022: Use findOrAllocOne() overload with ItemTypeId")
  ItemInternal* findOrAllocOne(Int64 uid,ItemTypeInfo* type, mesh::MeshInfos& mesh_info, bool& is_alloc);

  //! Allocates an element in the family and updates the corresponding \a mesh_info
  virtual Item allocOne(Int64 uid,ItemTypeId type_id, mesh::MeshInfos& mesh_info) =0;
  virtual Item findOrAllocOne(Int64 uid,ItemTypeId type_id, mesh::MeshInfos& mesh_info, bool& is_alloc) = 0;
  virtual IItemFamily* family() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* IITEMFAMILYMODIFIER_H_ */

// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemFamilyModifier.h                                       (C) 2000-2025 */
/*                                                                           */
/* Interface de modification d'une famille                                   */
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
 * \brief Interface de modification d'une famille.
 *
 * Cette classe permet de modifier génériquement les items d'une famille. Elle
 * est utilisée dans le workflow de modification des familles à partir du graphe
 * des dépendances (IItemFamilyNetwork).
 * Elle n'est implémentée que dans les familles d'items du maillage (ie tout sauf
 * ParticleFamily et DoFFamily, où elle n'est pas nécessaire)
 */
class ARCANE_CORE_EXPORT IItemFamilyModifier
{
 public:

  /** Destructeur de la classe */
  virtual ~IItemFamilyModifier() {}

 public:

  // DEPRECATED
  ARCANE_DEPRECATED_REASON("Y2022: Use allocOne() overload with ItemTypeId")
  ItemInternal* allocOne(Int64 uid,ItemTypeInfo* type, mesh::MeshInfos& mesh_info);
  // DEPRECATED
  ARCANE_DEPRECATED_REASON("Y2022: Use findOrAllocOne() overload with ItemTypeId")
  ItemInternal* findOrAllocOne(Int64 uid,ItemTypeInfo* type, mesh::MeshInfos& mesh_info, bool& is_alloc);

  //! Alloue un élément dans la famille et met à jour le \a mesh_info correspondant
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

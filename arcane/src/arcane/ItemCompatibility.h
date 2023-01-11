// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemCompatibility.h                                         (C) 2000-2023 */
/*                                                                           */
/* Méthodes assurant la compatibilité entre versions de Item.                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMCOMPATIBILITY_H
#define ARCANE_ITEMCOMPATIBILITY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// NOTE: Ce fichier est inclus directement par 'Item.h' et ne doit pas
// être inclus directement par d'autres fichiers.

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

namespace Materials
{
  class ComponentItemInternal;
}
namespace geometric
{
  class GeomShapeView;
}
namespace mesh
{
  class CellMerger;
  class ItemFamily;
  class ItemTools;
  class FaceFamily;
  class OneMeshItemAdder;
  class ParallelAMRConsistency;
  class MeshRefinement;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Méthodes pour conversions entre différentes classes de gestion
 * des entités
 *
 * Cette classe est temporaire et interne à Arcane. Seules les classes 'friend'
 * peuvent l'utiliser.
 */
class ItemCompatibility
{
  // Pour accéder à _internal()
  friend Materials::ComponentItemInternal;
  friend class ItemSharedInfo;
  friend class IItemFamilyModifier;
  friend geometric::GeomShapeView;
  friend class TotalviewAdapter;
  friend mesh::CellMerger;
  friend mesh::ItemFamily;
  friend mesh::ItemTools;
  friend mesh::FaceFamily;
  friend mesh::ParallelAMRConsistency;
  friend mesh::OneMeshItemAdder;
  friend mesh::MeshRefinement;

 private:

  static ItemInternal* _itemInternal(const Item& item)
  {
    return item._internal();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

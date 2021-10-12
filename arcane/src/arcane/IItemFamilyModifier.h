// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemFamilyModifier.h                                       (C) 2000-2017 */
/*                                                                           */
/* Interface de modification d'une famille                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IITEMFAMILYMODIFIER_H_ 
#define ARCANE_IITEMFAMILYMODIFIER_H_ 
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

#include "arcane/mesh/MeshInfos.h"

class ItemTypeInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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

  //! Alloue un élément dans la famille et met à jour le \a mesh_info correspondant
  virtual ItemInternal* allocOne(Int64 uid,ItemTypeInfo* type, mesh::MeshInfos& mesh_info) =0;
  virtual ItemInternal* findOrAllocOne(Int64 uid,ItemTypeInfo* type, mesh::MeshInfos& mesh_info, bool& is_alloc) = 0;
  virtual IItemFamily*  family() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* IITEMFAMILYMODIFIER_H_ */

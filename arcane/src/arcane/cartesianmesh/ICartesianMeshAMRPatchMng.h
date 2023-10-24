// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICartesianMeshAMRPatchMng.h                                 (C) 2000-2023 */
/*                                                                           */
/* Interface de gestionnaire de l'AMR par patch d'un maillage cartésien.     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CARTESIANMESH_ICARTESIANMESHAMRPATCHMNG_H
#define ARCANE_CARTESIANMESH_ICARTESIANMESHAMRPATCHMNG_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"
#include "arcane/Item.h"
#include "arcane/VariableTypedef.h"
#include "arcane/ItemEnumerator.h"

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"
#include "arcane/cartesianmesh/CartesianItemDirectionInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CARTESIANMESH_EXPORT ICartesianMeshAMRPatchMng
{
 public:

  ~ICartesianMeshAMRPatchMng() = default;

  /*!
   * @brief Méthode permettant de définir les mailles à raffiner.
   * @param cells_lids Les localIds des mailles.
   */
  virtual void flagCellToRefine(Int32ConstArrayView cells_lids) =0;

  /*!
   * @brief Méthode permettant de raffiner les mailles avec le
   * flag "II_Refine".
   */
  virtual void refine() =0;

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //ARCANE_CARTESIANMESH_ICARTESIANMESHAMRPATCHMNG_H

// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICartesianMeshAMRPatchMng.h                                 (C) 2000-2025 */
/*                                                                           */
/* Interface de gestionnaire de l'AMR par patch d'un maillage cartésien.     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CARTESIANMESH_ICARTESIANMESHAMRPATCHMNG_H
#define ARCANE_CARTESIANMESH_ICARTESIANMESHAMRPATCHMNG_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"
#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CARTESIANMESH_EXPORT ICartesianMeshAMRPatchMng
{
 public:

  virtual ~ICartesianMeshAMRPatchMng() = default;

 public:

  /*!
   * \brief Méthode permettant de définir les mailles à raffiner (à marquer avec le flag "II_Refine").
   * \param cells_lids Les localIds des mailles.
   * \param clear_old_flags Doit-on retirer les flags des raffinements/déraffinements précédents ?
   */
  virtual void flagCellToRefine(Int32ConstArrayView cells_lids, bool clear_old_flags) = 0;

  /*!
   * \brief Méthode permettant de définir les mailles à déraffiner (à marquer avec le flag "II_Coarsen").
   * \param cells_lids Les localIds des mailles.
   * \param clear_old_flags Doit-on retirer les flags des raffinements/déraffinements précédents ?
   */
  virtual void flagCellToCoarsen(Int32ConstArrayView cells_lids, bool clear_old_flags) = 0;

  /*!
   * \brief Méthode permettant de raffiner les mailles avec le
   * flag "II_Refine".
   */
  virtual void refine() = 0;

  /*!
   * \brief Méthode permettant de déraffiner les mailles de niveau 0.
   *
   * Un niveau de maille -1 sera créé avec des mailles parentes aux mailles
   * de niveau 0 puis tous les niveaux seront incrémentés de 1. Le niveau créé
   * par cette méthode sera donc le nouveau niveau 0.
   */
  virtual void createSubLevel() = 0;

  /*!
   * \brief Méthode permettant de retirer les mailles marquées avec le
   * flag "II_Coarsen".
   *
   * Les propriétaires des faces et des noeuds ayant des mailles marquées
   * et des mailles non marquées sont susceptibles d'être mis à jour.
   *
   * \param update_parent_flag Si true, alors les flags des parents seront
   * mis à jour. Cela inclut l'activation des mailles parentes.
   */
  virtual void coarsen(bool update_parent_flag) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //ARCANE_CARTESIANMESH_ICARTESIANMESHAMRPATCHMNG_H

﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemFlags.h                                                 (C) 2000-2024 */
/*                                                                           */
/* Drapeaux contenant les caractéristiques d'une entité.                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMFLAGS_H
#define ARCANE_CORE_ITEMFLAGS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Flags pour les caractéristiques des entités.
 */
class ARCANE_CORE_EXPORT ItemFlags
{
 public:

  using FlagType = Int32;

 public:

  // L'affichage 'lisible' des flags est implémenté dans ItemPrinter
  // Il doit être updaté si des flags sont ici changés

  enum : FlagType
  {
    II_Boundary = 1 << 1, //!< L'entité est sur la frontière
    II_HasFrontCell = 1 << 2, //!< L'entité a une maille devant
    II_HasBackCell = 1 << 3, //!< L'entité a une maille derrière
    II_FrontCellIsFirst = 1 << 4, //!< La première maille de l'entité est la maille devant
    II_BackCellIsFirst = 1 << 5, //!< La première maille de l'entité est la maille derrière
    II_Own = 1 << 6, //!< L'entité est une entité propre au sous-domaine
    II_Added = 1 << 7, //!< L'entité vient d'être ajoutée
    II_Suppressed = 1 << 8, //!< L'entité vient d'être supprimée
    II_Shared = 1 << 9, //!< L'entité est partagée par un autre sous-domaine
    II_SubDomainBoundary = 1 << 10, //!< L'entité est à la frontière de deux sous-domaines
    //II_JustRemoved = 1 << 11, //!< L'entité vient d'être supprimé
    II_JustAdded = 1 << 12, //!< L'entité vient d'être ajoutée
    II_NeedRemove = 1 << 13, //!< L'entité doit être supprimé
    II_SlaveFace = 1 << 14, //!< L'entité est une face esclave d'une interface
    II_MasterFace = 1 << 15, //!< L'entité est une face maître d'une interface
    II_Detached = 1 << 16, //!< L'entité est détachée du maillage
    II_HasTrace = 1 << 17, //!< L'entité est marquée pour trace (pour débug)

    II_Coarsen = 1 << 18, //!<  L'entité est marquée pour déraffinement
    II_DoNothing = 1 << 19, //!<  L'entité est bloquée
    II_Refine = 1 << 20, //!<  L'entité est marquée pour raffinement
    II_JustRefined = 1 << 21, //!<  L'entité vient d'être raffinée
    II_JustCoarsened = 1 << 22, //!<  L'entité vient d'être déraffiné
    II_Inactive = 1 << 23, //!<  L'entité est inactive //COARSEN_INACTIVE,
    II_CoarsenInactive = 1 << 24, //!<  L'entité est inactive et a des enfants tagués pour déraffinement

    II_UserMark1 = 1 << 25, //!< Marque utilisateur old_value 1<<24
    II_UserMark2 = 1 << 26 //!< Marque utilisateur  old_value 1<<25
  };

  static const int II_InterfaceFlags = II_Boundary + II_HasFrontCell + II_HasBackCell +
  II_FrontCellIsFirst + II_BackCellIsFirst;

  static constexpr bool isOwn(FlagType f) { return (f & II_Own) != 0; }
  static constexpr bool isShared(FlagType f) { return (f & II_Shared) != 0; }
  static constexpr bool isBoundary(FlagType f) { return (f & II_Boundary) != 0; }
  static constexpr bool isSubDomainBoundary(FlagType f) { return (f & II_Boundary) != 0; }
  static constexpr bool hasBackCell(FlagType f) { return (f & II_HasBackCell) != 0; }
  static constexpr bool isSubDomainBoundaryOutside(FlagType f)
  {
    return isSubDomainBoundary(f) && hasBackCell(f);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

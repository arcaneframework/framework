// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemFlags.h                                                 (C) 2000-2026 */
/*                                                                           */
/* Flags containing the characteristics of an entity.                        */
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
 * \brief Flags for entity characteristics.
 *
 * These flags allow storing information about entities (Item).
 * They are reserved for %Arcane and must not be used outside
 * of Arcane. The only exception concerns the values ItemFlags::II_UserMark1
 * and ItemFlags::II_UserMark2. They can be associated with entities via
 * the methods MutableItemBase::addFlags(), MutableItemBase::removeFlags() or
 * ItemBase::hasFlags().
 */
class ARCANE_CORE_EXPORT ItemFlags
{
 public:

  using FlagType = Int32;

 public:

  // The 'readable' display of the flags is implemented in ItemPrinter.
  // It must be updated if values here are modified or added.

  enum : FlagType
  {
    II_Boundary = 1 << 1, //!< The entity is on the boundary
    II_HasFrontCell = 1 << 2, //!< The entity has a front cell
    II_HasBackCell = 1 << 3, //!< The entity has a back cell
    II_FrontCellIsFirst = 1 << 4, //!< The first cell of the entity is the front cell
    II_BackCellIsFirst = 1 << 5, //!< The first cell of the entity is the back cell
    II_Own = 1 << 6, //!< The entity is a domain-specific entity
    II_Added = 1 << 7, //!< The entity has just been added
    II_Suppressed = 1 << 8, //!< The entity has just been suppressed
    II_Shared = 1 << 9, //!< The entity is shared by another subdomain
    II_SubDomainBoundary = 1 << 10, //!< The entity is at the boundary of two subdomains
    //II_JustRemoved = 1 << 11, //!< The entity has just been removed
    II_JustAdded = 1 << 12, //!< The entity has just been added
    II_NeedRemove = 1 << 13, //!< The entity must be removed
    II_SlaveFace = 1 << 14, //!< The entity is a slave face of an interface
    II_MasterFace = 1 << 15, //!< The entity is a master face of an interface
    II_Detached = 1 << 16, //!< The entity is detached from the mesh
    /*
     * \brief The entity uses edges instead of faces.
     *
     * This is only used for 2D cells of non-manifold meshes. If set, it means that
     * 1D entities are of type Edge and not of type Face.
     */
    II_HasEdgeFor1DItems = 1 << 17,

    II_Coarsen = 1 << 18, //!< The entity is marked for coarsening
    II_DoNothing = 1 << 19, //!< The entity is blocked
    II_Refine = 1 << 20, //!< The entity is marked for refinement
    II_JustRefined = 1 << 21, //!< The entity has just been refined
    II_JustCoarsened = 1 << 22, //!< The entity has just been coarsened
    II_Inactive = 1 << 23, //!< The entity is inactive //COARSEN_INACTIVE,
    II_CoarsenInactive = 1 << 24, //!< The entity is inactive and has children tagged for coarsening

    /*!
     * \brief [AMR Patch] The entity is marked as overlapping with
     * at least one AMR patch.
     */
    II_Overlap = 1 << 25,

    /*!
     * \brief [AMR Patch] The entity is marked as being in an AMR patch.
     */
    II_InPatch = 1 << 26,

    II_UserMark1 = 1 << 30, //!< User mark
    II_UserMark2 = 1 << 31 //!< User mark
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

  /*!
   * \brief Index in the face for the back cell.
   *
   * \retval -1 if there is no cell behind.
   * \retval 0 or 1 for the index of the back cell.
   *
   * If the index is positive, it is possible to retrieve
   * the back cell via Face::cell(ItemFlags::backCellIndex(f)).
   */
  static constexpr Int32 backCellIndex(FlagType f)
  {
    if (f & II_HasBackCell)
      return (f & II_BackCellIsFirst) ? 0 : 1;
    return -1;
  }

  /*!
   * \brief Index in the face for the front cell.
   *
   * \retval -1 if there is no cell in front.
   * \retval 0 or 1 for the index of the front cell.
   *
   * If the index is positive, it is possible to retrieve
   * the front cell via Face::cell(ItemFlags::frontCellIndex(f)).
   */
  static constexpr Int32 frontCellIndex(FlagType f)
  {
    if (f & II_HasFrontCell)
      return (f & II_FrontCellIsFirst) ? 0 : 1;
    return -1;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

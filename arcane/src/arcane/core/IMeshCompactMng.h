// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshCompactMng.h                                           (C) 2000-2025 */
/*                                                                           */
/* Interface for managing the compaction of mesh families.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHCOMPACTMNG_H
#define ARCANE_CORE_IMESHCOMPACTMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMeshCompacter;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface for managing the compaction of mesh families.
 *
 * While a compaction is in progress, it is forbidden to perform certain
 * operations on the mesh, such as creating a new family
 * or adding groups.
 *
 * The start of a compaction is done by calling beginCompact(). Once
 * the compaction is finished, you must call endCompact() to destroy
 * the IMeshCompacter instance.
 *
 * For more information on compaction, refer to the documentation
 * of IMeshCompacter.
 */
class ARCANE_CORE_EXPORT IMeshCompactMng
{
 public:

  virtual ~IMeshCompactMng() {} //<! Releases resources

 public:

  //! Associated mesh
  virtual IMesh* mesh() const = 0;

  /*!
   * \brief Starts a compaction on all families of the mesh.
   *
   * \pre compacter()==nullptr.
   */
  virtual IMeshCompacter* beginCompact() = 0;

  /*!
   * \brief Starts a compaction for the entity family \a family
   *
   * \pre compacter()==nullptr.
   *
   */
  virtual IMeshCompacter* beginCompact(IItemFamily* family) = 0;

  /*!
   * \brief Signals that the compaction is finished.
   *
   * This allows the deallocation of structures associated with the compaction.
   * \post exchanger()==nullptr.
   */
  virtual void endCompact() = 0;

  /*!
   * \brief Current active compacter.
   *
   * The compacter is non-null only if we are between a beginCompact()
   * and an endCompact()
   */
  virtual IMeshCompacter* compacter() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

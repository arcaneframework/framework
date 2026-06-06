// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IExtraGhostParticlesBuilder.h                               (C) 2000-2025 */
/*                                                                           */
/* Interface of a builder for "extraordinary" ghost meshes                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IEXTRAGHOSTPARTICLESBUILDER_H
#define ARCANE_CORE_IEXTRAGHOSTPARTICLESBUILDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a builder for "extraordinary" ghost meshes.
 *
 * An "extraordinary" ghost mesh is a ghost mesh added to the
 * ghost meshes defined by the mesh connectivity. Specifically,
 * the calculation of extraordinary ghost meshes is performed during every
 * mesh update or load balancing.
 *
 * \note makes the \a remove_old_ghost parameter of the IMesh::endUpdate()
 * method obsolete.
 *
 */
class IExtraGhostParticlesBuilder
{
 public:
  
  virtual ~IExtraGhostParticlesBuilder() {} //!< Releases resources.
  
 public:

  /*!
   * \brief Calculation of "extraordinary" meshes to send.
   *
   * Performs the calculation of "extraordinary" meshes following
   * a construction algorithm.
   */
  virtual void computeExtraParticlesToSend() =0;

  /*!
   * \brief Local indices of "extraordinary" meshes for sending
   *
   * Retrieves the array of "extraordinary" meshes destined for
   * the sub-domain \a rank.
   */
  virtual Int32ConstArrayView extraParticlesToSend(const String& family_name,Int32 rank) const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

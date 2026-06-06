// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IParticleFamily.h                                           (C) 2000-2025 */
/*                                                                           */
/* Interface of a particle family.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IPARTICLEFAMILY_H
#define ARCANE_CORE_IPARTICLEFAMILY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Mesh
 * \brief Interface of a particle family.
 *
 * A particle family is a family of entities (IItemFamily).
 * This interface only contains methods specific to particles.
 * For generic entity operations, you must go through
 * the IItemFamily interface via itemFamily().
 *
 * There can be several particle families per mesh.
 * Unlike classical mesh entities (node, cell, ...),
 * particles can be created directly.
 *
 */
class ARCANE_CORE_EXPORT IParticleFamily
{
 public:

  virtual ~IParticleFamily() {} //<! Releases resources

 public:

  virtual void build() = 0;

  //! Sets the flag to manage ghost particles for the family.
  virtual void setEnableGhostItems(bool value) = 0;

  //! Retrieves the flag to manage ghost particles for the family.
  virtual bool getEnableGhostItems() const = 0;

 public:

  //! Name of the family.
  virtual String name() const = 0;

  //! Full name of the family (including the mesh name).
  virtual String fullName() const = 0;

  //! Number of entities.
  virtual Integer nbItem() const = 0;

  //! Group of all particles.
  virtual ItemGroup allItems() const = 0;

 public:

  /*!
   * \brief Allocates particles.
   *
   * Allocates particles whose uniqueId() are given by the
   * array \a unique_ids.
   *
   * After calling this operation, you must call endUpdate() to notify
   * the instance that modifications are finished. It is possible to chain several
   * allocations before calling endUpdate(). Note that the returned view
   * may be invalidated after calling endUpdate() if compression is active.
   * \a items_local_id must have the same number of elements as unique_ids.
   */
  virtual ParticleVectorView addParticles(Int64ConstArrayView unique_ids,
                                          Int32ArrayView items_local_id) = 0;
  virtual ParticleVectorView addParticles2(Int64ConstArrayView unique_ids,
                                           Int32ConstArrayView owners,
                                           Int32ArrayView items) = 0;

  /*!
   * \brief Allocates particles in cells.
   *
   * This method is similar to addParticles() but allows specifying
   * directly the cells in which the particles will be created.
   */
  virtual ParticleVectorView addParticles(Int64ConstArrayView unique_ids,
                                          Int32ConstArrayView cells_local_id,
                                          Int32ArrayView items_local_id) = 0;

  virtual void removeParticles(Int32ConstArrayView items_local_id) = 0;

  /*!
   * \sa IItemFamily::endUpdate().
   */
  virtual void endUpdate() = 0;

  /*!
   * \brief Moves the particle \a particle into the cell \a new_cell.
   */
  virtual void setParticleCell(Particle particle, Cell new_cell) = 0;

  /*!
   * \brief Moves the list of particles \a particles into the new cells \a new_cells.
   */
  virtual void setParticlesCell(ParticleVectorView particles, CellVectorView new_cells) = 0;

 public:

  /*!
   * \brief Exchanging entities.
   *
   * This method is only supported for particle families.
   * For mesh elements such as nodes, faces, or cells, you must use IMesh::exchangeItems().
   *
   * The new owners of the entities are given by itemsNewOwner().
   *
   * This operation is blocking and collective.
   */
  virtual void exchangeParticles() = 0;

 public:

  //! Interface on the family.
  virtual IItemFamily* itemFamily() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

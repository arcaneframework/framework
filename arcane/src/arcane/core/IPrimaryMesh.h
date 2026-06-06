// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IPrimaryMesh.h                                              (C) 2000-2025 */
/*                                                                           */
/* Interface for the geometry of a mesh.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IPRIMARYMESH_H
#define ARCANE_CORE_IPRIMARYMESH_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/ItemTypes.h"
#include "arcane/core/VariableTypedef.h"
#include "arcane/core/IMesh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IParallelMng;
class IParticleExchanger;
class XmlNode;
class IMeshUtilities;
class IMeshModifier;
class Properties;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//INFO: The complete documentation is in Mesh.dox
class IPrimaryMesh
: public IMesh
{
 public:

  virtual ~IPrimaryMesh() {} //<! Releases resources

 public:

  //! Node coordinates
  virtual VariableNodeReal3& nodesCoordinates() =0;

  /*!
   * \brief Positions the mesh dimension (1D, 2D, or 3D).
   *
   * The dimension corresponds to the dimension of the mesh elements (Cell).
   * If meshes of multiple dimensions are present, the highest dimension must be indicated.
   *
   * The dimension must be set before allocating meshes if one
   * uses allocateCells(), and must not be modified afterward.
   */
  virtual void setDimension(Integer dim) =0;  

  /*! \brief Reloads the mesh from protected variables
   */
  virtual void reloadMesh() =0;

  //NOTE: Complete documentation for this method is in Mesh.dox
  //! Allocation of a mesh.
  virtual void allocateCells(Integer nb_cell,Int64ConstArrayView cells_infos,bool one_alloc=true) =0;

  /*!
   * \brief Indicates the end of mesh allocation.
   *
   * As long as this method has not been called, it is not valid to use this
   * instance, except for allocating the mesh (allocateCells()).
   *
   * This method is collective.
   */
  virtual void endAllocate() =0;

  /*!
   * \brief Deallocates the mesh.
   *
   * This deletes all entities and all entity groups.
   * The mesh must then be reallocated via the call to allocateCells().
   * This call also deletes the mesh dimension, which must
   * be reset by setDimension(). It is therefore possible to change the
   * mesh dimension afterward.
   *
   * This method is collective.
   *
   * \warning This method is experimental and many edge effects are
   * possible. Notably, the current implementation does not support deallocation
   * when there are partial variables on the mesh.
   */
  virtual void deallocate() =0;

  /*!
   * \brief Specific initial allocator.
   *
   * If null, allocateCells() must be used.
   */
  virtual IMeshInitialAllocator* initialAllocator() { return nullptr; }

 public:

  /*!
   * \brief Variable containing the identifier of the owning subdomain.
   *
   Returns the variable containing the identifier of the owning subdomain
   of entities of the given kind.
   
   \warning This variable is used for generating synchronization messages
   between subdomains and must not
   be modified.
   */
  virtual VariableItemInt32& itemsNewOwner(eItemKind kind) =0;

  //! Changes the owning subdomains of entities
  virtual void exchangeItems() =0;

 public:

  /*!
   * \internal
   * \brief Positions entity owners based on the mesh owner.
   *
   Positions the owners of entities other than meshes (Node, Edge, and Face)
   based on the mesh owner. This operation is only useful
   in parallel and must only be called during initialization after
   the endAllocate() method.
   *
   This operation is collective.
   */
  virtual void setOwnersFromCells() =0;

  /*!
   * \internal
   * \brief Positions partitioning information.
   */
  virtual void setMeshPartInfo(const MeshPartInfo& mpi) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IPrimaryMesh.h                                              (C) 2000-2016 */
/*                                                                           */
/* Interface de la géométrie d'un maillage.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IPRIMARYMESH_H
#define ARCANE_IPRIMARYMESH_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"
#include "arcane/ItemTypes.h"
#include "arcane/VariableTypedef.h"
#include "arcane/IMesh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

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
//INFO: La doc complete est dans Mesh.dox
class IPrimaryMesh
: public IMesh
{
 public:

  virtual ~IPrimaryMesh() {} //<! Libère les ressources

 public:

  //! Coordonnées des noeuds
  virtual VariableNodeReal3& nodesCoordinates() =0;

  /*!
   * \brief Positionne la dimension du maillage (1D, 2D ou 3D).
   *
   * La dimension correspond à la dimension des éléments mailles (Cell).
   * Si des mailles de plusieurs dimensions sont présentes, il faut indiquer
   * la dimension la plus importante.
   *
   * La dimension doit être positionnée avant d'allouer des mailles, et ne
   * doit plus être modifiée ensuite.
   */
  virtual void setDimension(Integer dim) =0;  

  /*! \brief Recharge le maillage à partir des variables protégées
   */
  virtual void reloadMesh() =0;

  //! Allocation d'un maillage.
  virtual void allocateCells(Integer nb_cell,Int64ConstArrayView cells_infos,bool one_alloc=true) =0;

  /*!
   * \brief Indique une fin d'allocation de mailles.
   *
   * Tant que cette méthode n'a pas été appelée, il n'est pas valide d'utiliser cette
   * instance, sauf pour allouer le maillage (allocateCells()).
   *
   * Cette méthode est collective.
   */
  virtual void endAllocate() =0;

 public:

  /*!
   * \brief Variable contenant l'identifiant du sous-domaine propriétaire.
   *
   Retourne la variable contenant l'identifiant du sous-domaine propriétaire
   des entités du genre \a kind.
   
   \warning Cette variable est utilisée pour la fabrication des messages
   de synchronisation entre sous-domaines et ne doit pas
   être modifiée.
   */
  virtual VariableItemInt32& itemsNewOwner(eItemKind kind) =0;

  //! Change les sous-domaines propriétaires des entités
  virtual void exchangeItems() =0;

 public:

  /*!
   * \internal
   * \brief Positionne les propriétaires des entités à partir du propriétaire des mailles.
   *
   * Positionne les propriétaires des entités autres que les mailles (Node,Edge et Face)
   * en se basant sur le propriétaire aux mailles. Cette opération n'a d'intéret
   * qu'en parallèle et ne doit être appelée que lors de l'initialisation après
   * la méthode endAllocate().
   *
   * Cette opération est collective.
   */
  virtual void setOwnersFromCells() =0;

  /*!
   * \internal
   * \brief Positionne les informations de partitionnement.
   */
  virtual void setMeshPartInfo(const MeshPartInfo& mpi) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

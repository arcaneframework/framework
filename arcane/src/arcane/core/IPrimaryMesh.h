// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IPrimaryMesh.h                                              (C) 2000-2023 */
/*                                                                           */
/* Interface de la géométrie d'un maillage.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IPRIMARYMESH_H
#define ARCANE_IPRIMARYMESH_H
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
   * La dimension doit être positionnée avant d'allouer des mailles si on
   * utilise allocateCells(), et ne doit plus être modifiée ensuite.
   */
  virtual void setDimension(Integer dim) =0;  

  /*! \brief Recharge le maillage à partir des variables protégées
   */
  virtual void reloadMesh() =0;

  //NOTE: Documentation complète de cette méthode dans Mesh.dox
  //!Allocation d'un maillage.
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

  /*!
   * \brief Désalloue le maillage.
   *
   * Cela supprime toutes les entités et tous les groupes d'entités.
   * Le maillage devra ensuite être alloué à nouveau via l'appel à allocateCells().
   * Cet appel supprime aussi la dimension du maillage qui devra
   * être repositionné par setDimension(). Il est donc possible de changer la
   * dimension du maillage par la suite.
   *
   * Cette méthode est collective.
   *
   * \warning Cette méthode est expérimentale et de nombreux effets de bords sont
   * possibles. Notamment, l'implémentation actuelle ne supporte pas la désallocation
   * lorsqu'il y a des variables partielles sur le maillage.
   */
  virtual void deallocate() =0;

  /*!
   * \brief Allocateur initial spécifique.
   *
   * Si nul, il faut utiliser allocateCells().
   */
  virtual IMeshInitialAllocator* initialAllocator() { return nullptr; }

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

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

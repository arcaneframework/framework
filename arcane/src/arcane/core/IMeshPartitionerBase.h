// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshPartitioner.h                                          (C) 2000-2025 */
/*                                                                           */
/* Interface d'un partitionneur de maillage.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHPARTITIONERBASE_H
#define ARCANE_CORE_IMESHPARTITIONERBASE_H
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
 * \brief Interface d'un partitionneur de maillage.
 */
class IMeshPartitionerBase
{
 public:

  virtual ~IMeshPartitionerBase() = default; //!< Libère les ressources.

 public:

  /*!
   * Re-partitionne le maillage \a mesh
   *
   * Cette méthode change les propriétaires des entités et
   * remplit la variable IItemFamily::itemsNewOwner() de chaque famille d'entité
   * du maillage \a mesh avec le numéro du nouveau sous-domaine propriétaire.
   *
   * \note Cette méthode est réservée aux développeurs Arcane.
   * Si un module souhaite effectuer un re-partitionnement,
   * il faut appeler la méthode 
   * IMeshUtilities::partitionAndExchangeMeshWithReplication()
   * qui gère à la fois le partitionnement et l'échange des
   * informations et supporte la réplication de domaine.
   */
  virtual void partitionMesh(bool initial_partition) =0;

  //! Maillage associé
  virtual IPrimaryMesh* primaryMesh() =0;

  //! Notification lors de la fin d'un re-partitionnement (après échange des entités)
  virtual void notifyEndPartition() =0;
  };

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

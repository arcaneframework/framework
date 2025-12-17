// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IParallelReplicationInfo.h                                  (C) 2000-2025 */
/*                                                                           */
/* Informations sur la réplication de sous-domaines.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IPARALLELREPLICATION_H
#define ARCANE_CORE_IPARALLELREPLICATION_H
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
 * \ingroup Parallel
 * \brief Informations sur la réplication des sous-domaines en parallèle.
 */
class ARCANE_CORE_EXPORT IParallelReplication
{
 public:

  virtual ~IParallelReplication() = default;

 public:

  //! Indique si la réplication est active
  virtual bool hasReplication() const = 0;

  //! Nombre de réplication
  virtual Int32 nbReplication() const = 0;

  //! Rang dans la réplication (de 0 à nbReplication()-1)
  virtual Int32 replicationRank() const = 0;

  /*!
   * \brief Indique si ce rang de réplication est le maître.
   *
   * Cela est utile par exemple pour les sorties, afin qu'un seul
   * réplicat ne sorte les informations.
   */
  virtual bool isMasterRank() const = 0;

  //! Rang dans la réplication du maître.
  virtual Int32 masterReplicationRank() const = 0;

  /*!
   * \brief Communicateur associé à tous les réplicats représentant un même sous-domaine.
   *
   * Vaut 0 s'il n'y a pas de réplication (hasReplication() est faux).
   */
  virtual IParallelMng* replicaParallelMng() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

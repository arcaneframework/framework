// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IParallelReplicationInfo.h                                  (C) 2000-2012 */
/*                                                                           */
/* Informations sur la réplication de sous-domaines.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IPARALLELREPLICATION_H
#define ARCANE_IPARALLELREPLICATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IParallelMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Parallel
 * \brief Informations sur la réplication des sous-domaines en parallèle.
 */
class ARCANE_CORE_EXPORT IParallelReplication
{
 public:

  virtual ~IParallelReplication(){}

 public:

  //! Indique si la réplication est active
  virtual bool hasReplication() const =0;

  //! Nombre de réplication
  virtual Int32 nbReplication() const =0;

  //! Rang dans la réplication (de 0 à nbReplication()-1)
  virtual Int32 replicationRank() const =0;

  /*!
   * \brief Indique si ce rang de réplication est le maître.
   *
   * Cela est utile par exemple pour les sorties, afin qu'un seul
   * réplicat ne sorte les informations.
   */
  virtual bool isMasterRank() const =0;

  //! Rang dans la réplication du maître.
  virtual Int32 masterReplicationRank() const =0;

  /*!
   * \brief Communicateur associé à tous les réplicats représentant un même sous-domaine.
   *
   * Vaut 0 s'il n'y a pas de réplication (hasReplication() est faux).
   */
  virtual IParallelMng* replicaParallelMng() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

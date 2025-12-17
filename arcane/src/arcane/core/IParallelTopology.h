// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IParallelTopology.h                                         (C) 2000-20255 */
/*                                                                           */
/* Informations sur la topologie d'allocation des coeurs de calcul.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IPARALLELTOPOLOGY_H
#define ARCANE_CORE_IPARALLELTOPOLOGY_H
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
 * \brief Informations sur la topologie d'allocation des coeurs de calcul.
 *
 * Une instance de cette classe est liée à un IParallelMng.
 *
 * Elle permet de savoir comment les rangs de ce IParallelMng sont alloués
 * sur le cluster et dans les processus.
 *
 */
class ARCANE_CORE_EXPORT IParallelTopology
{
 public:

  virtual ~IParallelTopology() = default; //!< Libère les ressources.

 public:

  //! Gestionnaire de parallélisme associé
  virtual IParallelMng* parallelMng() const = 0;

  //! Indique si ce rang est le rang maître pour une machine (noeud)
  virtual bool isMasterMachine() const = 0;

  //! Liste des rangs qui sont sur la même machine
  virtual Int32ConstArrayView machineRanks() const = 0;

  /*!
   * \brief Rang de cette instance dans la liste des machines (noeuds).
   *
   * Ce rang est compris entre 0 et masterMachineRanks().size().
   */
  virtual Int32 machineRank() const = 0;

  /*!
   * \brief Liste des rangs maîtres pour chaque machine (noeud).
   *
   * Cette liste est la même pour tous les rangs.
   */
  virtual Int32ConstArrayView masterMachineRanks() const = 0;

  //! Indique si ce rang est le maitre dans les rangs de ce processus.
  virtual bool isMasterProcess() const = 0;

  //! Liste des rangs qui sont dans le même processus (en multi-threading)
  virtual Int32ConstArrayView processRanks() const = 0;

  /*!
   * \brief Rang de cette instance dans la liste des processus.
   *
   * Ce rang est compris entre 0 et masterProcessRanks().size().
   */
  virtual Int32 processRank() const = 0;

  /*!
   * \brief Liste des rangs maitres pour chaque processus.
   *
   * Cette liste est la même pour tous les rangs.
   */
  virtual Int32ConstArrayView masterProcessRanks() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

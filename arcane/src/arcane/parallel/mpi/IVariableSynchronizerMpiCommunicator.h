// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IVariableSynchronizerMpiCommunicator.h                      (C) 2000-2022 */
/*                                                                           */
/* Interface d'un communicateur MPI spécifique pour les synchronisations.    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_MPI_IVARIABLESYNCHRONIZERMPICOMMUNICATOR_H
#define ARCANE_PARALLEL_MPI_IVARIABLESYNCHRONIZERMPICOMMUNICATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/parallel/mpi/ArcaneMpi.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class VariableSynchronizer;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un communicateur MPI spécifique pour les synchronisations.
 *
 * Ce communicateur permet d'utiliser les méthodes de MPI 3.1 telles
 * que MPI_Neighbor_alltoallv() pour les synchronisations.
 *
 * Il faut appeler compute() avant de pouvoir utiliser ce communicateur
 * spécifique.
 */
class ARCANE_MPI_EXPORT IVariableSynchronizerMpiCommunicator
{
 public:

  virtual ~IVariableSynchronizerMpiCommunicator() = default;

 public:

  //! Calcul le communicateur spécifique
  virtual void compute(VariableSynchronizer* var_syncer) = 0;

  /*!
   * \brief Récupère le communicateur spécifique de la topologie.
   *
   * Ce communicateur ne doit pas être conservé car il peut être invalidé
   * entre deux appels à compute().
   */
  virtual MPI_Comm communicator() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

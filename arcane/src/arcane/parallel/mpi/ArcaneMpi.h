// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneMpi.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Déclarations globales pour la partie MPI de Arcane.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_MPI_ARCANEMPI_H
#define ARCANE_PARALLEL_MPI_ARCANEMPI_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"
#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCANE_MPIOP_CALL ARCCORE_MPIOP_CALL

#define ARCANE_MPI_HAS_NONBLOCKINGCOLLECTIVE

namespace Arcane
{
using MessagePassing::Mpi::IMpiReduceOperator;
using MessagePassing::Mpi::MpiAdapter;
using MessagePassing::Mpi::MpiDatatype;
using MessagePassing::Mpi::MpiLock;
using MessagePassing::Mpi::MpiSerializeDispatcher;
using MessagePassing::Mpi::StdMpiReduceOperator;
namespace MpiBuiltIn = MessagePassing::Mpi::MpiBuiltIn;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Mécanisme pour initialiser automatiquement MPI et les
 * services d'échange de messages de Arcane en fonction des paramètres
 * de ApplicationBuildInfo.
 */
extern "C" ARCANE_MPI_EXPORT void
arcaneAutoDetectMessagePassingServiceMPI();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Indique si le runtime actuel de MPI a le support de CUDA.
 *
 * Si l'implémentation MPI supporte CUDA cela permet à MPI d'accéder
 * directement à la mémoire du GPU.
 */
extern "C++" ARCANE_MPI_EXPORT bool
arcaneIsCudaAwareMPI();

/*!
 * \brief Indique si le runtime actuel de MPI a le support des accélérateurs.
 *
 * Si l'implémentation MPI supporte CUDA ou HIP cela permet à MPI d'accéder
 * directement à la mémoire du GPU.
 */
extern "C++" ARCANE_MPI_EXPORT bool
arcaneIsAcceleratorAwareMPI();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! \internal
extern "C++" ARCANE_MPI_EXPORT void
arcaneInitializeMPI(int* argc, char*** argv, int wanted_thread_level);

//! \internal
extern "C++" ARCANE_MPI_EXPORT void
arcaneFinalizeMPI();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  


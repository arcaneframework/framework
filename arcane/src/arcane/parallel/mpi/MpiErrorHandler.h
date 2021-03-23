// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiErrorHandler.h                                           (C) 2000-2020 */
/*                                                                           */
/* Gestionnaire d'erreur pour MPI.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_MPI_MPIERRORHANDLER_H
#define ARCANE_PARALLEL_MPI_MPIERRORHANDLER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/parallel/mpi/ArcaneMpi.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface des messages pour le type \a Type
 */
class ARCANE_MPI_EXPORT MpiErrorHandler
{
 public:
  
  MpiErrorHandler();
  ~MpiErrorHandler();
  void removeHandler();
  void registerHandler(MPI_Comm comm);

 private:

  MPI_Errhandler m_err_handler;
  bool m_has_err_handler;

 private:

  // Handler d'erreur
  static void _ErrorHandler(MPI_Comm *, int *, ...);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

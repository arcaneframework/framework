// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiErrorHandler.cc                                          (C) 2000-2020 */
/*                                                                           */
/* Gestionnaire d'erreur pour MPI.                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/parallel/mpi/MpiErrorHandler.h"

#include "arcane/utils/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiErrorHandler::
MpiErrorHandler()
: m_err_handler(MPI_Errhandler())
, m_has_err_handler(false)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiErrorHandler::
~MpiErrorHandler()
{
  removeHandler();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiErrorHandler::
removeHandler()
{
  if (m_has_err_handler){
    MPI_Errhandler_free(&m_err_handler);
    m_has_err_handler = false;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiErrorHandler::
registerHandler(MPI_Comm comm)
{
  if (m_has_err_handler)
    ARCANE_FATAL("Handler already registered");

  // Regarder s'il faut le rendre optionnel.
  MPI_Comm_create_errhandler(&MpiErrorHandler::_ErrorHandler,&m_err_handler);
  MPI_Comm_set_errhandler(comm,m_err_handler);
  m_has_err_handler = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiErrorHandler::
_ErrorHandler(MPI_Comm* comm, int* error_code, ...)
{
  ARCANE_UNUSED(comm);

  char error_buf[MPI_MAX_ERROR_STRING+1];
  int error_len = 0;
  int e = *error_code;
  // int MPI_Error_string(int errorcode, char *string, int *resultlen);
  MPI_Error_string(e,error_buf,&error_len);
  error_buf[error_len] = '\0';
  error_buf[MPI_MAX_ERROR_STRING] = '\0';
  
  // int MPI_Error_class(int errorcode, int *errorclass);

  ARCANE_FATAL("Error in MPI call code={0} msg={1}",*error_code,error_buf);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

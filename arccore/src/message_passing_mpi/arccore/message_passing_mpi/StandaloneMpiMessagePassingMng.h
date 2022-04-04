﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StandaloneMpiMessagePassingMng.h                            (C) 2000-2020 */
/*                                                                           */
/* Version autonome de MpiMessagePassingMng.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_STANDALONEMPIMESSAGEPASSINGMNG_H
#define ARCCORE_MESSAGEPASSINGMPI_STANDALONEMPIMESSAGEPASSINGMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MpiMessagePassingMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Version autonome de MpiMessagePassingMng.
 *
 * La création se fait via la méthode statique create().
 */
class ARCCORE_MESSAGEPASSINGMPI_EXPORT StandaloneMpiMessagePassingMng
: public MpiMessagePassingMng
{
  class Impl;

 private:

  StandaloneMpiMessagePassingMng(Impl* p);

 public:

  ~StandaloneMpiMessagePassingMng() override;

 public:

  //! Créé un gestionnaire associé au communicateur \a comm.
  static MpiMessagePassingMng* create(MPI_Comm comm, bool clean_comm=false);

 private:

  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

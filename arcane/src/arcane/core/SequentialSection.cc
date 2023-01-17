// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SequentialSection.cc                                        (C) 2000-2015 */
/*                                                                           */
/* Section du code à exécuter séquentiellement.                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/ParallelFatalErrorException.h"

#include "arcane/IParallelMng.h"
#include "arcane/ISubDomain.h"
#include "arcane/SequentialSection.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SequentialSection::
SequentialSection(IParallelMng* pm)
: m_parallel_mng(pm)
, m_has_error(false)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SequentialSection::
SequentialSection(ISubDomain* sd)
: m_parallel_mng(sd->parallelMng())
, m_has_error(false)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SequentialSection::
~SequentialSection() ARCANE_NOEXCEPT_FALSE
{
  Integer sid = m_parallel_mng->commRank();
  if (sid==0){
    Integer error_value = m_has_error ? 1 : 0;
    //cerr << "** ERROR SEND " << error_value << "!\n";
    ArrayView<Integer> iv(1,&error_value);
    m_parallel_mng->broadcast(iv,0);
    if (m_has_error)
      _sendError();
  }
  else{
    return;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SequentialSection::
_init()
{
  Integer sid = m_parallel_mng->commRank();
  if (sid==0){
    return;
  }
  else{
    Integer error_value = 0;
    ArrayView<Integer> iv(1,&error_value);
    m_parallel_mng->broadcast(iv,0);
    //cerr << "** ERROR RECV " << sid << ' ' << error_value << "!\n";
   if (error_value!=0)
      _sendError();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SequentialSection::
_sendError()
{
  Integer sid = m_parallel_mng->commRank();
  ITraceMng* trace = m_parallel_mng->traceMng();
  if (trace)
    trace->logdate() << "Subdomain " << sid << "Fatal in sequantial section";
  m_parallel_mng->barrier();
  throw ParallelFatalErrorException("sequential section");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SequentialSection::
setError(bool is_error)
{
  m_has_error = is_error;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


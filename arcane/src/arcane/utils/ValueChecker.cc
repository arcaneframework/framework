// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ValueChecker.cc                                             (C) 2000-2016 */
/*                                                                           */
/* Fonctions pour convertir un type en un autre.                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ValueChecker.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ValueChecker::
_addError(const String& message)
{
  ++m_nb_error;
  if (m_throw_on_error){
    throw FatalErrorException(m_trace_info,message);
  }
  else{
    m_last_error_str = message;
    m_ostr() << message << '\n';
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ValueChecker::
throwIfError()
{
  if (m_nb_error!=0){
    m_nb_error = 0;
    throw FatalErrorException(m_trace_info,m_ostr.str());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


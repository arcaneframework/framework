// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BadAlignmentException.cc                                    (C) 2000-2016 */
/*                                                                           */
/* Exception lorsqu'une adresse n'est pas correctement alignée.              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/String.h"
#include "arcane/utils/BadAlignmentException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BadAlignmentException::
BadAlignmentException(const String& awhere,const void* ptr,Integer alignment)
: Exception("BadAlignmentException",awhere)
, m_ptr(ptr)
, m_wanted_alignment(alignment)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BadAlignmentException::
BadAlignmentException(const TraceInfo& awhere,const void* ptr,Integer alignment)
: Exception("BadAlignmentException",awhere)
, m_ptr(ptr)
, m_wanted_alignment(alignment)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BadAlignmentException::
explain(std::ostream& m) const
{
  Int64 alignment = 0;
  if (m_wanted_alignment>0){
    Int64 ptr = (Int64)m_ptr;
    alignment = ptr % m_wanted_alignment;
  }
	m << "Bad alignment for address " << m_ptr
    << " alignment=" << alignment
    << " (wanted=" << m_wanted_alignment << ").";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GoBackwardException.cc                                      (C) 2000-2015 */
/*                                                                           */
/* Exception pour demander un retour-arrière de la boucle en temps           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/String.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/GoBackwardException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GoBackwardException::
GoBackwardException(const String& where)
: Exception("GoBackward",where)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GoBackwardException::
GoBackwardException(const String& where,const String& message)
: Exception("GoBackward",where)
, m_message(message)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GoBackwardException::
GoBackwardException(const TraceInfo& where)
: Exception("GoBackward",where)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GoBackwardException::
GoBackwardException(const TraceInfo& where,const String& message)
: Exception("GoBackward",where)
, m_message(message)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GoBackwardException::
explain(std::ostream& m) const
{
	m << "Go backward.";
  if (!m_message.null())
		m << "Message: " << m_message << '\n';
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


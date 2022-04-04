// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GoBackwardException.h                                       (C) 2000-2015 */
/*                                                                           */
/* Exception pour demander un retour-arrière de la boucle en temps           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_GOBACKWARDEXCEPTION_H
#define ARCANE_UTILS_GOBACKWARDEXCEPTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Exception.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Exception pour demander un retour-arrière de la boucle en temps
 */
class ARCANE_UTILS_EXPORT GoBackwardException
: public Exception
{
 public:
	
  GoBackwardException(const String& where);
  GoBackwardException(const String& where,const String& message);
  GoBackwardException(const TraceInfo& where);
  GoBackwardException(const TraceInfo& where,const String& message);
  ~GoBackwardException() ARCANE_NOEXCEPT {}

 public:
	
  virtual void explain(std::ostream& m) const;

 private:

	String m_message;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  


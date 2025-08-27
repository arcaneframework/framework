// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BadVariantTypeException.h                                   (C) 2000-2007 */
/*                                                                           */
/* Exception levée lorsqu'un variant n'est pas du type souhaité              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_DATATYPE_BADVARIANTTYPEEXCEPTION_H
#define ARCANE_DATATYPE_BADVARIANTTYPEEXCEPTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Exception.h"

#include "arcane/core/datatype/VariantBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 *
 * \brief Exception sur un type de variant non valide.
 *
 * Cette exception est envoyée lorsqu'on essaye de construire un variant
 * avec un type inconnu.
 */
class BadVariantTypeException
: public Exception
{
 public:
	
  BadVariantTypeException(const String& where,VariantBase::eType wrongType);
  
 public:

  virtual void explain(std::ostream& m) const;

 private:

  VariantBase::eType m_wrong_type;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif


// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BadAlignmentException.h                                     (C) 2000-2017 */
/*                                                                           */
/* Exception lorsqu'une adresse n'est pas correctement alignée.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_BADCASTEXCEPTION_H
#define ARCANE_UTILS_BADCASTEXCEPTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Exception.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Core
 * \brief Exception lorsqu'une adresse n'est pas correctement alignée.
 */
class ARCANE_UTILS_EXPORT BadAlignmentException
: public Exception
{
 public:
	
  BadAlignmentException(const String& where,const void* ptr,Integer alignment);
  BadAlignmentException(const TraceInfo& where,const void* ptr,Integer alignment);

  virtual void explain(std::ostream& m) const;

 private:

  const void* m_ptr;
  Integer m_wanted_alignment;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  


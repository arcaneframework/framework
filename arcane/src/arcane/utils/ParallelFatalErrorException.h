// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelFatalErrorException.h                               (C) 2000-2018 */
/*                                                                           */
/* Exception lorsqu'une erreur fatale 'parallèle' est survenue.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_PARALLELFATALERROREXCEPTION_H
#define ARCANE_UTILS_PARALLELFATALERROREXCEPTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Exception.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Parallel
 * \brief Exception lorsqu'une erreur fatale 'parallèle' est générée.
 *
 * Une erreur fatale 'parallèle' est une erreur fatale commune à tout les
 * sous-domaines. Dans ce cas, il est possible d'arrêter proprement le code
 */
class ARCANE_UTILS_EXPORT ParallelFatalErrorException
: public Exception
{
 public:
	
  ParallelFatalErrorException(const String& where);
  ParallelFatalErrorException(const TraceInfo& where);
  ParallelFatalErrorException(const String& where,const String& message);
  ParallelFatalErrorException(const TraceInfo& where,const String& message);
  ParallelFatalErrorException(const ParallelFatalErrorException& rhs)
  : Exception(rhs){}
  ~ParallelFatalErrorException() ARCANE_NOEXCEPT {}

 public:
	
  virtual void explain(std::ostream& m) const;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  


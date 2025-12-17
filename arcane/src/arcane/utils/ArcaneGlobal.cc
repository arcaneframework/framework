// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneGlobal.cc                                             (C) 2000-2025 */
/*                                                                           */
/* Déclarations générales de Arcane.                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IndexOutOfRangeException.h"
#include "arcane/utils/ArithmeticException.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/BadAlignmentException.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/ArraySimdPadder.h"

#include "arcane/utils/Iostream.h"
#include "arcane/utils/IMemoryInfo.h"

// Ces includes ne sont pas utilisés par ce fichier
// mais il faut au moins les lire une fois sous Windows
// pour que les symboles externes soient créés
#include "arcane/utils/IFunctor.h"
#include "arcane/utils/IFunctorWithAddress.h"
#include "arcane/utils/IRangeFunctor.h"
#include "arcane/utils/SpinLock.h"
#include "arcane/utils/IPerformanceCounterService.h"
#include "arcane/utils/IProfilingService.h"
#include "arcane/utils/DataTypeContainer.h"
#include "arcane/utils/ITraceMngPolicy.h"
#include "arcane/utils/IThreadImplementationService.h"
#include "arcane/utils/IMessagePassingProfilingService.h"
#include "arcane/utils/ISymbolizerService.h"
#include "arcane/utils/IDataCompressor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file utils/ArcaneGlobal.h
 *
 * \brief Fichier de configuration d'Arcane.
 */
/*!
 * \namespace Arcane
 *
 * \brief Espace de nom d'Arcane.
 *
 * Toutes les classes et types utilisés dans \b Arcane sont dans ce
 * namespace.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT void
arcaneRangeError(Int32 i,Int32 max_size)
{
  arcaneDebugPause("arcaneRangeError");
  throw IndexOutOfRangeException(A_FUNCINFO,String(),i,0,max_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT void
arcaneRangeError(Int64 i,Int64 max_size)
{
  arcaneDebugPause("arcaneRangeError");
  throw IndexOutOfRangeException(A_FUNCINFO,String(),i,0,max_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT void
_internalArcaneMathError(long double value,const char* funcname)
{
  cerr << "** FATAL: Argument error for a mathematical operation:\n";
  cerr << "** FATAL: Argument: " << value << '\n';
  if (funcname)
    cerr << "** FATAL: Operation: " << funcname << '\n';
  arcaneDebugPause("arcaneMathError");
  throw ArithmeticException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT void
_internalArcaneMathError(long double value1,long double value2,const char* funcname)
{
  cerr << "** FATAL: Argument error for a mathematical operation:\n";
  cerr << "** FATAL: Argument1: " << value1 << '\n';
  cerr << "** FATAL: Argument2: " << value2 << '\n';
  if (funcname)
    cerr << "** FATAL: Operation: " << funcname << '\n';
  arcaneDebugPause("arcaneMathError");
  throw ArithmeticException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT void
arcaneNotYetImplemented(const char* file,const char* func,
                        unsigned long line,const char* text)
{
  cerr << file << ':' << func << ':' << line << '\n';
  cerr << "sorry, functionality not yet implemented";
  if (text)
    cerr << ": " << text;
  cerr << '\n';
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT void
arcaneNullPointerError()
{
  cerr << "** FATAL: null pointer.\n";
  cerr << "** FATAL: Trying to dereference a null pointer.\n";
  arcaneDebugPause("arcaneNullPointerPtr");
  throw FatalErrorException(A_FUNCINFO,"null pointer");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT void
arcaneThrowNullPointerError(const char* ptr_name,const char* text)
{
  throw FatalErrorException(A_FUNCINFO,text ? text : ptr_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT Integer
arcaneCheckArraySize(unsigned long long size)
{
  ARCANE_THROW_IF((size>=ARCANE_INTEGER_MAX),ArgumentException,"value '{0}' too big for Array size",size);
  return static_cast<Integer>(size);
}

extern "C++" ARCANE_UTILS_EXPORT Integer
arcaneCheckArraySize(long long size)
{
  ARCANE_THROW_IF( (size>=ARCANE_INTEGER_MAX),ArgumentException,"value '{0}' too big for Array size",size);
  ARCANE_THROW_IF((size<0),ArgumentException,"invalid negative value '{0}' for Array size",size);
  return static_cast<Integer>(size);
}

extern "C++" ARCANE_UTILS_EXPORT Integer
arcaneCheckArraySize(unsigned long size)
{
  ARCANE_THROW_IF((size>=ARCANE_INTEGER_MAX),ArgumentException,"value '{0}' too big for Array size",size);
  return static_cast<Integer>(size);
}

extern "C++" ARCANE_UTILS_EXPORT Integer
arcaneCheckArraySize(long size)
{
  ARCANE_THROW_IF((size>=ARCANE_INTEGER_MAX),ArgumentException,"value '{0}' too big for Array size",size);
  ARCANE_THROW_IF((size<0),ArgumentException,"invalid negative value '{0}' for Array size",size);
  return static_cast<Integer>(size);
}

extern "C++" ARCANE_UTILS_EXPORT Integer
arcaneCheckArraySize(unsigned int size)
{
  ARCANE_THROW_IF((size>=ARCANE_INTEGER_MAX),ArgumentException,"value '{0}' too big for Array size",size);
  return static_cast<Integer>(size);
}

extern "C++" ARCANE_UTILS_EXPORT Integer
arcaneCheckArraySize(int size)
{
  ARCANE_THROW_IF((size>=ARCANE_INTEGER_MAX),ArgumentException,"value '{0}' too big for Array size",size);
  ARCANE_THROW_IF((size<0),ArgumentException,"invalid negative value '{0}' for Array size",size);
  return static_cast<Integer>(size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT void
arcaneCheckAlignment(const void* ptr,Integer alignment)
{
  if (alignment<=0)
    return;
  Int64 iptr = (intptr_t)ptr;
  Int64 modulo = iptr % alignment;
  if (modulo!=0)
    throw BadAlignmentException(A_FUNCINFO,ptr,alignment);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT Integer
arcaneSizeWithPadding(Integer size)
{
  return ArraySimdPadder::getSizeWithPadding(size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \defgroup Collection Collection
 *
 * \brief Classes collections.
 *
 * Les collections sont les classes %Arcane qui gèrent un ensemble d'objet.
 * En général, la collection la plus utilisée est le tableau. Pour
 * plus de renseignements, se reporter à la page \ref arcanedoc_core_types_array_usage.
 */

/*!
 * \defgroup Module Module
 *
 * \brief Classes des modules
 */

/*!
 * \defgroup Core Core
 *
 * \brief Classes du noyau
 */

/*!
 * \defgroup CaseOption CaseOption
 *
 * \brief Classes du jeu de données
 */

/*!
 * \defgroup Mesh Mesh
 *
 * \brief Classes du maillage
 */

/*!
 * \defgroup Variable Variable
 *
 * \brief Classes liées aux variables
 */

/*!
 * \defgroup Parallel Parallel
 *
 * \brief Classes liées aux parallélisme
 */

/*!
 * \defgroup Xml Xml
 *
 * \brief Classes liées à XML.
 */

/*!
 * \defgroup StandardService StandardService
 *
 * \brief Services standards
 */

/*!
 * \defgroup IO IO
 *
 * \brief Gestion des entrées/sorties
 */

/*!
 * \defgroup Math Math
 *
 * \brief Fonctions mathématiques
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

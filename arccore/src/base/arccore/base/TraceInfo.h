// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TraceInfo.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Informations de trace.                                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_TRACEINFO_H
#define ARCCORE_BASE_TRACEINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations de trace.
 */
class TraceInfo
{
 public:

  constexpr TraceInfo()
  : m_file("(None)")
  , m_name("(None)")
  , m_line(-1)
  , m_print_signature(true)
  {}
  constexpr TraceInfo(const char* afile, const char* func_name, int aline)
  : m_file(afile)
  , m_name(func_name)
  , m_line(aline)
  , m_print_signature(true)
  {}
  constexpr TraceInfo(const char* afile, const char* func_name, int aline, bool print_signature)
  : m_file(afile)
  , m_name(func_name)
  , m_line(aline)
  , m_print_signature(print_signature)
  {}

 public:

  constexpr const char* file() const { return m_file; }
  constexpr int line() const { return m_line; }
  constexpr const char* name() const { return m_name; }
  constexpr bool printSignature() const { return m_print_signature; }

 private:

  const char* m_file;
  const char* m_name;
  int m_line;
  bool m_print_signature;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT
std::ostream& operator<<(std::ostream& o,const TraceInfo&);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef __GNUG__
#define A_FUNCINFO \
::Arccore::TraceInfo(__FILE__,__PRETTY_FUNCTION__,__LINE__)
#define A_FUNCNAME \
::Arccore::TraceInfo(__FILE__,__PRETTY_FUNCTION__,__LINE__,false)
#else
// Normalement valide uniquement avec extension c99
#ifdef ARCCORE_OS_WIN32
#define A_FUNCINFO \
::Arccore::TraceInfo(__FILE__,__FUNCTION__,__LINE__)
#define A_FUNCNAME \
::Arccore::TraceInfo(__FILE__,__FUNCTION__,__LINE__,false)
#else
#define A_FUNCINFO \
::Arccore::TraceInfo(__FILE__,__func__,__LINE__)
#define A_FUNCNAME \
::Arccore::TraceInfo(__FILE__,__func__,__LINE__,false)
#endif
#endif

#define A_FUNCINFO1(name)\
::Arccore::TraceInfo(__FILE__,name,__LINE__)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  


// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TraceAccessor2.h                                            (C) 2000-2020 */
/*                                                                           */
/* Traces.                                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_TRACEACCESSOR2_H
#define ARCANE_UTILS_TRACEACCESSOR2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::internal
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief API EN COURS DE CONCEPTION. NE PAS UTILISER.
 */
class ARCANE_UTILS_EXPORT TraceAccessor2
: public TraceAccessor
{
 public:
  explicit TraceAccessor2(ITraceMng* tm) : TraceAccessor(tm){}
 public:
  bool isActive() const { return m_is_active; }
 private:
  bool m_is_active = false;
};

template<typename T>
class TracePrinter
{
 public:
  TracePrinter(const char* name,const T& r) : m_name(name), m_r(r){}
  friend std::ostream& operator<<(std::ostream& o,const TracePrinter<T>& x)
  {
    o << " " << x.m_name << "=" << x.m_r;
    return o;
  }
 public:
  const char* m_name;
  const T& m_r;
};

template<typename T>
inline TracePrinter<T> makeTracePrinter(const char* name,const T& field_name)
{
  return TracePrinter<T>(name,field_name);
}

template<typename ...Args> String
format2(const String& str,const Args& ...args)
{
  return String::format(str,args...);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline bool
isTraceActive(const TraceAccessor2* tr)
{
  return tr->isActive();
}

#define A_TR(field_name) Arcane::internal::makeTracePrinter(#field_name,field_name)

#define A_TR2(name,field_name) Arcane::internal::makeTracePrinter(name,field_name)

#define A_INFO(...)\
  do {\
    if (Arcane::internal::isTraceActive(this)) {  \
      info() << Arcane::String::format(__VA_ARGS__);  \
    }\
  } while(false)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

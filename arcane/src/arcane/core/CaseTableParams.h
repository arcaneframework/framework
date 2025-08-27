// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseTableParams.h                                           (C) 2000-2025 */
/*                                                                           */
/* Paramètres d'une table de marche du jeu de données.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_CASETABLEPARAMS_H
#define ARCANE_CORE_CASETABLEPARAMS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/CaseTable.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Paramètre d'une fonction.
 */
class CaseTableParams
{
 public:

  class Impl;

 public:

  CaseTableParams(CaseTable::eParamType v);
  virtual ~CaseTableParams();

 public:

  bool null() const;
  Integer nbElement() const;
  void value(Integer id, Real& v) const;
  void value(Integer id, Integer& v) const;
  CaseTable::eError appendValue(const String& value);
  CaseTable::eError setValue(Integer id, const String& value);
  CaseTable::eError setValue(Integer id, Real v);
  CaseTable::eError setValue(Integer id, Integer v);
  void removeValue(Integer id);
  void toString(Integer id, String& str) const;
  void setType(ICaseFunction::eParamType new_type);

  void getRange(Real v, Int32& begin, Int32& end) const;
  void getRange(Integer v, Int32& begin, Int32& end) const;

 private:

  Impl* m_p;

 private:

  template <typename T> inline void _getRange(T v, Int32& begin, Int32& end) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  





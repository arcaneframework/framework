// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseTableParams.h                                           (C) 2000-2008 */
/*                                                                           */
/* Paramètres d'une table de marche du jeu de données.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CASETABLEPARAMS_H
#define ARCANE_CASETABLEPARAMS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#include "arcane/CaseTable.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

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
  void value(Integer id,Real& v) const;
  void value(Integer id,Integer& v) const;
  void value(Integer id,bool& v) const;
  CaseTable::eError appendValue(const String& value);
  CaseTable::eError setValue(Integer id,const String& value);
  CaseTable::eError setValue(Integer id,Real v);
  CaseTable::eError setValue(Integer id,Integer v);
  CaseTable::eError setValue(Integer id,bool v);
  void removeValue(Integer id);
  void toString(Integer id,String& str) const;
  void setType(ICaseFunction::eParamType new_type);
 private:
  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  





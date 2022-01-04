// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseDocumentLangTranslator.h                                (C) 2000-2017 */
/*                                                                           */
/* Classe gérant la traduction d'un jeu de données dans une autre langue.    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_CASEDOCUMENTLANGTRANSLATOR_H
#define ARCANE_IMPL_CASEDOCUMENTLANGTRANSLATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ICaseMng;
class XmlNode;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe gérant la traduction d'un jeu de données dans une autre langue.
 */
class ARCANE_IMPL_EXPORT CaseDocumentLangTranslator
: public TraceAccessor
{
 public:

  CaseDocumentLangTranslator(ITraceMng* tm);
  virtual ~CaseDocumentLangTranslator();

  virtual void build();

 public:

  String translate(ICaseMng* cm,const String& new_lang);

 private:

  String m_global_convert_string;
 private:
  void _addConvert(XmlNode node,const String& new_name);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

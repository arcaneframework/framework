// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseDocumentLangTranslator.h                                (C) 2000-2017 */
/*                                                                           */
/* Class managing the translation of a dataset into another language.        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_CASEDOCUMENTLANGTRANSLATOR_H
#define ARCANE_IMPL_CASEDOCUMENTLANGTRANSLATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ICaseMng;
class XmlNode;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class managing the translation of a dataset into another language.
 */
class ARCANE_IMPL_EXPORT CaseDocumentLangTranslator
: public TraceAccessor
{
 public:

  CaseDocumentLangTranslator(ITraceMng* tm);
  virtual ~CaseDocumentLangTranslator();

  virtual void build();

 public:

  String translate(ICaseMng* cm, const String& new_lang);

 private:

  String m_global_convert_string;

 private:

  void _addConvert(XmlNode node, const String& new_name);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptionsMulti.h                                          (C) 2000-2019 */
/*                                                                           */
/* Options du jeu de données gérant plusieurs occurences.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CASEOPTIONSMULTI_H
#define ARCANE_CASEOPTIONSMULTI_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/CaseOptions.h"
#include "arcane/utils/UniqueArray.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe de base d'un tableau d'options complexes du jeu de données.
 */
class ARCANE_CORE_EXPORT CaseOptionsMulti
: public CaseOptions
, public ICaseOptionsMulti
{
 public:
	
  CaseOptionsMulti(ICaseMng*,const String& tag_root_name,
                   const XmlNode& element,Integer min_occurs,Integer max_occurs);
  CaseOptionsMulti(ICaseOptionList*,const String& tag_root_name,
                   const XmlNode& element,Integer min_occurs,Integer max_occurs);

 public:

  ICaseOptions* toCaseOptions() override { return this; }
  void addChild(ICaseOptionList* v) override { m_values.add(v); }
  Integer nbChildren() const override { return m_values.size(); }
  ICaseOptionList* child(Integer index) const override { return m_values[index]; }
  ICaseOptionsMulti* toCaseOptionsMulti() { return this; }

 private:

  UniqueArray<ICaseOptionList*> m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

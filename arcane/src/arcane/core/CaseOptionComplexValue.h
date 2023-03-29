// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptionComplexValue.h                                    (C) 2000-2023 */
/*                                                                           */
/* Option du jeu de données de type 'complexe'.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CASEOPTIONCOMPLEXVALUE_H
#define ARCANE_CASEOPTIONCOMPLEXVALUE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounter.h"

#include "arcane/core/ICaseOptionList.h"
#include "arcane/core/XmlNode.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class ICaseOptionsMulti;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe de base d'une valeur d'une option complexe.
 *
 * Une option complexe est composé de plusieurs instances de cette classe.
 */
class ARCANE_CORE_EXPORT CaseOptionComplexValue
{
 public:

  CaseOptionComplexValue(ICaseOptionsMulti* opt,ICaseOptionList* clist,const XmlNode& parent_elem);
  virtual ~CaseOptionComplexValue();

 public:

  ARCANE_DEPRECATED_REASON("Y2022: Do not access XML item from option")
  XmlNode element() const { return m_element; }

  ARCANE_DEPRECATED_REASON("Y2022: This method is internal to Arcane. Do not use it")
  ICaseOptionList* configList() const { return m_config_list.get(); }

  //! Nom complet au format donné par la norme XPath.
  String xpathFullName() const { return m_element.xpathFullName(); }

 protected:

  // Les deux méthodes suivantes sont utilisés par le générateur 'axl2cc' et
  // ne doivent pas être modifiées.
  ICaseOptionList* _configList() { return m_config_list.get(); }
  XmlNode _element() { return m_element; }

 private:

  ReferenceCounter<ICaseOptionList> m_config_list;
  XmlNode m_element;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

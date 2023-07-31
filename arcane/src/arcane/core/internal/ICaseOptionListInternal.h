// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICaseOptionListInternal.h                                   (C) 2000-2023 */
/*                                                                           */
/* Partie interne à Arcane de 'ICaseOptionList'.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ICASEOPTIONLISTINTERNAL_H
#define ARCANE_CORE_ICASEOPTIONLISTINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/CaseOptionTypes.h"
#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief API Interne de l'interface d'une liste d'options du jeu de données.
 */
class ARCANE_CORE_EXPORT ICaseOptionListInternal
{
 protected:

  virtual ~ICaseOptionListInternal() = default;

 public:

  virtual void addConfig(CaseOptionBase* o, const XmlNode& parent) = 0;

  //! Positionne l'élément racine de la liste, avec \a parent_element comme parent. Si déjà positionné, ne fait rien
  virtual void setRootElementWithParent(const XmlNode& parent_element) = 0;

  //! Positionne l'élément racine de la liste. Si déjà positionné, lance une exception
  virtual void setRootElement(const XmlNode& root_element) = 0;

  //! Ajoute les éléments fils ne correspondants par à de options dans \a nlist
  virtual void addInvalidChildren(XmlNodeList& nlist) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

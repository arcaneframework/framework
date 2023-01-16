﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICaseOptionList.h                                           (C) 2000-2019 */
/*                                                                           */
/* Options du jeu de donnés.                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ICASEOPTIONLIST_H
#define ARCANE_ICASEOPTIONLIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/CaseOptionTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'une liste d'options du jeu de données.
 */
class ARCANE_CORE_EXPORT ICaseOptionList
{
 protected:

  virtual ~ICaseOptionList() = default;

 public:
  /*!
   * \brief Retourne l'élément lié à cette liste d'option.
   * S'il n'y en a pas, retourne 0. S'il y en a plusieurs, retourne le
   * premier.
   */
  virtual XmlNode rootElement() const =0;
  //! Retourne l'élément parent.
  virtual XmlNode parentElement() const =0;

  //! Ajoute l'option \a o avec le parent \a parent
  virtual void addConfig(CaseOptionBase* o,XmlNode parent) =0;
  //! Ajoute la liste \a co à la liste des fils.
  virtual void addChild(ICaseOptions* co) =0;
  //! Supprime \a co de la liste des fils.
  virtual void removeChild(ICaseOptions* co) =0;
  //! Retourne le gestionnaire du cas
  virtual ICaseMng* caseMng() const =0;
  //! Positionne l'élément racine de la liste, avec \a parent_element comme parent. Si déjà positionné, ne fait rien
  virtual void setRootElementWithParent(XmlNode parent_element) =0;
  //! Positionne l'élément racine de la liste. Si déjà positionné, lance une exception
  virtual void setRootElement(XmlNode root_element) =0;
  //! Lis les valeurs des options à partir des éléments du DOM.
  virtual void readChildren(bool is_phase1) =0;
  //! Ajoute les éléments fils ne correspondants par à de options dans \a nlist
  virtual void addInvalidChildren(XmlNodeList& nlist) =0;
  //! Affiche la liste des options filles dans le langage \a lang et leur valeur
  virtual void printChildren(const String& lang,int indent) =0;
  //! Retourne le nom de l'élément de cette liste
  virtual String rootTagName() const =0;
  //! Ajoute à la liste \a col tous les options filles.
  virtual void deepGetChildren(Array<CaseOptionBase*>& col) =0;
  //! Indique si l'option est présente dans le jeu de données.
  virtual bool isPresent() const =0;
  //! Indique si l'option est optionnelle
  virtual bool isOptional() const =0;
  //! Applique le visiteur \a visitor
  virtual void visit(ICaseDocumentVisitor* visitor) =0;
  //! Nom complet au format XPath correspondant à rootElement()
  virtual String xpathFullName() const =0;
  //! Handle du maillage associé
  virtual MeshHandle meshHandle() const =0;
  //! Ajoute une référence
  virtual void addReference() =0;
  //! Supprime une référence
  virtual void removeReference() =0;

  /*!
   * \brief Désactive l'option comme si elle était absente.
   *
   * Cela est utilisé par exemple si l'option est associée à un maillage
   * qui n'est pas défini.
   */
  virtual void disable() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

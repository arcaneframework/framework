// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICaseOptionList.h                                           (C) 2000-2023 */
/*                                                                           */
/* Options du jeu de données.                                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ICASEOPTIONLIST_H
#define ARCANE_CORE_ICASEOPTIONLIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/CaseOptionTypes.h"
#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class XmlNode;
class ICaseMng;
class XmlNodeList;
class ICaseOptionListInternal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'une liste d'options du jeu de données.
 */
class ARCANE_CORE_EXPORT ICaseOptionList
{
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();

 protected:

  virtual ~ICaseOptionList() = default;

 public:

  /*!
   * \brief Retourne l'élément lié à cette liste d'option.
   *
   * S'il n'y en a pas, retourne un XmlNode nul. S'il y en a plusieurs, retourne le
   * premier.
   */
  virtual XmlNode rootElement() const =0;
  //! Retourne l'élément parent.
  virtual XmlNode parentElement() const =0;
  //! Ajoute la liste \a co à la liste des fils.
  virtual void addChild(ICaseOptions* co) =0;
  //! Supprime \a co de la liste des fils.
  virtual void removeChild(ICaseOptions* co) =0;
  //! Retourne le gestionnaire du cas
  virtual ICaseMng* caseMng() const =0;

  //! Lis les valeurs des options à partir des éléments du DOM.
  virtual void readChildren(bool is_phase1) =0;
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
  //! Nombre minimum d'occurences
  virtual Integer minOccurs() const = 0;
  //! Nombre maximum d'occurences
  virtual Integer maxOccurs() const = 0;
  //! Applique le visiteur \a visitor
  virtual void visit(ICaseDocumentVisitor* visitor) =0;
  //! Nom complet au format XPath correspondant à rootElement()
  virtual String xpathFullName() const =0;
  //! Handle du maillage associé
  virtual MeshHandle meshHandle() const =0;

  //! Document associé.
  virtual ICaseDocumentFragment* caseDocumentFragment() const =0;

  /*!
   * \brief Désactive l'option comme si elle était absente.
   *
   * Cela est utilisé par exemple si l'option est associée à un maillage
   * qui n'est pas défini.
   */
  virtual void disable() = 0;

 public:

  //! Ajoute l'option \a o avec le parent \a parent
  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  virtual void addConfig(CaseOptionBase* o,XmlNode parent) =0;

  //! Positionne l'élément racine de la liste, avec \a parent_element comme parent. Si déjà positionné, ne fait rien
  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  virtual void setRootElementWithParent(XmlNode parent_element) =0;

  //! Positionne l'élément racine de la liste. Si déjà positionné, lance une exception
  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  virtual void setRootElement(XmlNode root_element) =0;

  //! Ajoute les éléments fils ne correspondants par à de options dans \a nlist
  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  virtual void addInvalidChildren(XmlNodeList& nlist) =0;

 public:

  //! API interne à Arcane
  virtual ICaseOptionListInternal* _internalApi() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICaseDocument.h                                             (C) 2000-2023 */
/*                                                                           */
/* Interface d'une classe gérant un document XML du jeu de données.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ICASEDOCUMENT_H
#define ARCANE_ICASEDOCUMENT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/XmlNodeList.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CaseNodeNames;
class XmlNode;
class IXmlDocumentHolder;
class CaseOptionError;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'une partie d'un jeu de données.
 */
class ICaseDocumentFragment
{
 public:

  virtual ~ICaseDocumentFragment() = default;

 public:

  /*!
   * \brief Retourne le document Xml du jeu de données.
   * Ce pointeur reste la propriété de cette classe et est détruit lorsque cette
   * instance est détruite.
   */
  virtual IXmlDocumentHolder* documentHolder() =0;

  //! Retourne le noeud document
  virtual XmlNode documentNode() =0;

  //! Retourne l'élément racine.
  virtual XmlNode rootElement() =0;

  //! Langage utilisé dans le jeu de données
  virtual String language() const =0;

  //! Catégorie utilisée pour les valeurs par défaut.
  virtual String defaultCategory() const =0;

  //! Retourne l'instance contenant les noms des noeuds XML par langage.
  virtual CaseNodeNames* caseNodeNames() =0;
  
 public:

  //! Ajoute une erreur dans le jeu de données
  virtual void addError(const CaseOptionError& case_error) =0;

  //! Ajoute un avertissement dans le jeu de données
  virtual void addWarning(const CaseOptionError& case_error) =0;

  // Indique si le jeu de données contient des erreurs.
  virtual bool hasError() const =0;

  // Indique si le jeu de données contient des avertissements.
  virtual bool hasWarnings() const =0;

  //! Ecrit les erreurs dans le flot \a o
  virtual void printErrors(std::ostream& o) =0;

  //! Ecrit les avertissements dans le flot \a o
  virtual void printWarnings(std::ostream& o) =0;

  //! Supprime les messages d'erreurs et d'avertissements enregistrés
  virtual void clearErrorsAndWarnings() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'une classe gérant un document XML du jeu de données.
 * \todo Ne plus hériter de ICaseDocumentFragment (utiliser la méthode fragment() à la place)
 */
class ICaseDocument
: public ICaseDocumentFragment
{
 public:

  //! Construit l'instance
  virtual void build() =0;

  //! Clone le document
  virtual ICaseDocument* clone() =0;

 public:

  //! Retourne l'instance contenant les noms des noeuds XML par langage.
  virtual CaseNodeNames* caseNodeNames() =0;

  //! Retourne l'élément des informations pour Arcane
  virtual XmlNode arcaneElement() =0;

  //! Retourne l'élément des informations de configuration
  virtual XmlNode configurationElement() =0;

  //! Retourne l'élément contenant le choix de la boucle en temps
  virtual XmlNode timeloopElement() =0;
  //! Retourne l'élément contenant le titre du cas
  virtual XmlNode titleElement() =0;
  //! Retourne l'élément contenant la description du cas
  virtual XmlNode descriptionElement() =0;
  //! Retourne l'élément contenant la description des modules
  virtual XmlNode modulesElement() =0;
  //! Retourne l'élément contenant la description des services
  virtual XmlNode servicesElement() =0;

  //! Retourne l'élément racine des fonctions
  virtual XmlNode functionsElement() =0;

  //! Retourne l'élément racine des informations de maillage
  virtual const XmlNodeList& meshElements() =0;

  //! Elément contenant la liste des maillages (nouveau mécanisme) (peut être nul)
  virtual XmlNode meshesElement() =0;

  //! Nom de la classe d'utilisation du cas
  virtual String userClass() const =0;
  //! Positionne le nom de la classe d'utilisation du cas
  virtual void setUserClass(const String& value) =0;

  //! Nom du code du cas
  virtual String codeName() const =0;
  //! Positionne le nom du code du cas
  virtual void setCodeName(const String& value) =0;

  //! Numéro de version du code correspondant au cas
  virtual String codeVersion() const =0;
  //! Positionne le numéro de version du code
  virtual void setCodeVersion(const String& value) =0;

  //! Nom du système d'unité du document.
  virtual String codeUnitSystem() const =0;
  //! Positionne le nom du systmème d'unité du document.
  virtual void setCodeUnitSystem(const String& value) =0;

  //! Positionne la catégorie utilisée pour les valeurs par défaut.
  virtual void setDefaultCategory(const String& v) =0;

  //! Fragment correspondant à ce document
  virtual ICaseDocumentFragment* fragment() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

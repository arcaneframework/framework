// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptionBase.h                                            (C) 2000-2023 */
/*                                                                           */
/* Classe d'une base d'une option du jeu de donnés.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CASEOPTIONBASE_H
#define ARCANE_CASEOPTIONBASE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"

#include "arcane/XmlNode.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CaseOptionBuildInfo;
class ICaseDocument;
class ICaseDocumentFragment;
class ICaseMng;
class ICaseOptionList;
class ISubDomain;
class ICaseFunction;
class CaseOptionBasePrivate;
class ICaseDocumentVisitor;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base d'une option du jeu de donnée.
 *
 * \ingroup CaseOption
 *
 * Fait le lien entre l'option de nom \a m_name et le noeud du
 * DOM correspondant.
 */
class ARCANE_CORE_EXPORT CaseOptionBase
{
 protected:

  CaseOptionBase(const CaseOptionBuildInfo& cob);

 public:

  virtual ~CaseOptionBase();

 public:

  //! Retourne le vrai nom (non traduit) de l'option.
  String trueName() const;

  //! Retourne le nom de l'option correspondant au langage du jeu de données
  String name() const;

  //! Nom dans la langue \a lang de l'option. Retourne \a name() si pas de traduction.
  String translatedName(const String& lang) const;

  //! Récupère la valeur du fichier de configuration pour la variable
  void search(bool is_phase1);

  //! Imprime la valeur de l'option dans le langage \a lang,sur le flot \a o
  virtual void print(const String& lang,std::ostream& o) const =0;

  //! Gestionnaire de cas
  ICaseMng* caseMng() const;

  //! OptionList parent
  ICaseOptionList* parentOptionList() const;

  //! Gestionnaire de traces
  ITraceMng* traceMng() const;

  //! Gestionnaire de sous-domaine
  ARCCORE_DEPRECATED_2019("Do not use subDomain(). Try to get subDomain from an other way.")
  ISubDomain* subDomain() const;
  
  //! Retourne le gestionnaire de document
  ARCANE_DEPRECATED_REASON("Y2023: use caseMng()->caseDocument() instead.")
  ICaseDocument* caseDocument() const;
  
  //! Retourne le document associé à cette option
  ICaseDocumentFragment* caseDocumentFragment() const;

  //! Positionne l'élément racine à \a root_element
  void setRootElement(const XmlNode& root_element);

  //! Retourne l'élément racine du DOM
  XmlNode rootElement() const;
  
  //! Retourne la fonction liée à cette option ou `nullptr` s'il n'y en a pas
  virtual ICaseFunction* function() const =0;

  //! Nombre minimum d'occurences (pour une option multiple)
  Integer minOccurs() const;

  //! Nombre maximum d'occurences (pour une option multiple) (-1 == unbounded)
  Integer maxOccurs() const;

  //! Permet de savoir si une option est optionnelle.
  bool isOptional() const;

  /*! \brief Met à jour la valeur de l'option à partir d'une fonction.
   *
   * Si l'option n'est pas liée à une table de marche, ne fait rien.
   * Sinon, utilise \a current_time ou \a current_iteration suivant
   * le type de paramètre de la fonction pour calculer la nouvelle
   * valeur de l'option. Cette valeur sera ensuite accessible normalement par
   * la méthode operator().
   */
  virtual void updateFromFunction(Real current_time,Integer current_iteration) =0;

  /*!
    \brief Ajoute une traduction pour le nom de l'option.
    *
    Ajoute le nom \a name de l'option correspondant au langage \a lang.
    Si une traduction existe déjà pour ce langage, elle est remplacée par
    celle-ci.
  */
  void addAlternativeNodeName(const String& lang,const String& name);

  //! Ajoute la valeur par défaut \a value à la catégorie \a category
  void addDefaultValue(const String& category,const String& value);

  //! Applique le visiteur sur cette option
  virtual void visit(ICaseDocumentVisitor* visitor) const =0;

  //! Lève une exception si l'option n'a pas été initialisée.
  void checkIsInitialized() const { _checkIsInitialized(); }

 protected:

  //! Retourne la valeur par défaut de l'option ou 0 s'il n'y en a pas
  String _defaultValue() const;

  void _setDefaultValue(const String& def_value);

 protected:

  virtual void _search(bool is_phase1) =0;
  void _setIsInitialized();
  bool _isInitialized() const;
  void _checkIsInitialized() const;
  void _checkMinMaxOccurs(Integer nb_occur);
  String _xpathFullName() const;

 private:

  CaseOptionBasePrivate* m_p; //!< Implémentation.

 private:

  void _setTranslatedName();
  void _setCategoryDefaultValue();
  /*! \brief Constructeur de copie.
   *
   * Le constructeur par copie est privée car l'option ne doit pas être
   * copiée, notamment à cause du ICaseFunction qui est unique.
   */  
  CaseOptionBase(const CaseOptionBase& from) = delete;
  //! Opérateur de recopie
  CaseOptionBase& operator=(const CaseOptionBase& from) = delete;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

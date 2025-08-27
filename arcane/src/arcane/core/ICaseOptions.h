// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICaseOptions.h                                              (C) 2000-2025 */
/*                                                                           */
/* Options du jeu de donnés.                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ICASEOPTIONS_H
#define ARCANE_CORE_ICASEOPTIONS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/CaseOptionTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'une liste d'options du jeu de données.
 *
 * Cette interface est gérée par un compteur de référence et ne doit pas
 * être détruite explictement.
 */
class ARCANE_CORE_EXPORT ICaseOptions
{
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();

 protected:

  virtual ~ICaseOptions() = default;

 public:

  //! Vrai nom (non traduit) de l'élément.
  virtual String rootTagTrueName() const = 0;

  //! Nom de l'élément dans le langage du jeu de données.
  virtual String rootTagName() const = 0;

  //! Nom dans la langue \a lang de l'option. Retourne \a rootTagTrueName() si pas de traduction.
  virtual String translatedName(const String& lang) const = 0;

  ARCCORE_DEPRECATED_2019("Use read(eCaseOptionReadPhase) instead")
  virtual void read(bool is_phase1) = 0;

  /*!
   * \brief Effectue la lecture de la phase \a read_phase des options.
   */
  virtual void read(eCaseOptionReadPhase read_phase) = 0;

  virtual void addInvalidChildren(XmlNodeList&) = 0;

  virtual void printChildren(const String& lang, int indent) = 0;

  //! Retourne le service associé ou `nullptr` s'il n'y en a pas.
  virtual IServiceInfo* caseServiceInfo() const = 0;

  //! Retourne le module associé ou `nullptr` s'il n'y en a pas.
  virtual IModule* caseModule() const = 0;

  /*!
   * \internal
   * \brief Associe le service \a m à ce jeu de données.
   */
  virtual void setCaseServiceInfo(IServiceInfo* m) = 0;

  /*!
   * \internal
   * \brief  Associe le module \a m à ce jeu de données.
   */
  virtual void setCaseModule(IModule* m) = 0;

  /*!
   * \internal
   * \brief Ajoute à la liste \a col tous les options filles.
   */
  virtual void deepGetChildren(Array<CaseOptionBase*>& col) = 0;

  virtual ICaseOptionList* configList() = 0;

  virtual const ICaseOptionList* configList() const = 0;

  //! Fonction indiquant l'état d'activation de l'option
  virtual ICaseFunction* activateFunction() = 0;

  /*!
   * \brief Indique si l'option est présente dans le jeu de données.
   *
   * Une option peut ne pas apparaître si elle ne contient que des
   * options ayant une valeur par défaut.
   */
  virtual bool isPresent() const = 0;

  /*!
   * \internal
   * \brief Ajoute une traduction pour le nom de l'option.
   *
   * Ajoute le nom \a name de l'option correspondant au langage \a lang.
   * Si une traduction existe déjà pour ce langage, elle est remplacée par
   * celle-ci.
   */
  virtual void addAlternativeNodeName(const String& lang, const String& name) = 0;

  virtual ICaseMng* caseMng() const = 0;
  virtual ITraceMng* traceMng() const = 0;
  /*!
   * \brief Sous-domain associé.
   *
   * \deprecated Ne plus utiliser cette méthode car à terme une option
   * pourra exister sans sous-domaine.
   */
  ARCCORE_DEPRECATED_2019("Do not use subDomain(). Try to get subDomain from an other way.")
  virtual ISubDomain* subDomain() const = 0;
  ARCCORE_DEPRECATED_2019("Use meshHandle().mesh() instead")
  virtual IMesh* mesh() const = 0;
  virtual MeshHandle meshHandle() const = 0;
  ARCANE_DEPRECATED_REASON("Y2023: use caseMng()->caseDocument() instead.")
  virtual ICaseDocument* caseDocument() const = 0;
  virtual ICaseDocumentFragment* caseDocumentFragment() const = 0;

  /*!
   * \internal
   * Détache l'option de son parent.
   */
  virtual void detach() = 0;

  //! Applique le visiteur sur cette option
  virtual void visit(ICaseDocumentVisitor* visitor) const = 0;

  //! Nom complet au format XPath correspondant à rootElement()
  virtual String xpathFullName() const = 0;

 public:

  virtual Ref<ICaseOptions> toReference() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'une liste d'options présente plusieurs fois.
 */
class ARCANE_CORE_EXPORT ICaseOptionsMulti
{
 public:

  virtual ~ICaseOptionsMulti() {}

 public:

  virtual void multiAllocate(const XmlNodeList&) =0;
  virtual ICaseOptions* toCaseOptions() =0;
  virtual void addChild(ICaseOptionList* v) =0;
  virtual Integer nbChildren() const =0;
  virtual ICaseOptionList* child(Integer index) const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CORE_EXPORT ISubDomain*
_arcaneDeprecatedGetSubDomain(ICaseOptions* opt);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

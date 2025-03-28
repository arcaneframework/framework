// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptions.h                                               (C) 2000-2025 */
/*                                                                           */
/* Options du jeu de données.                                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CASEOPTIONS_H
#define ARCANE_CASEOPTIONS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounterImpl.h"

#include "arcane/utils/Array.h"
#include "arcane/utils/String.h"
#include "arcane/utils/Ref.h"

#include "arcane/core/XmlNode.h"
#include "arcane/core/ICaseOptions.h"
#include "arcane/core/ICaseOptionList.h"

// Ces fichiers ne sont pas nécessaires pour ce '.h' mais on les ajoute
// car axlstar inclus uniquement 'CaseOptions.h'
#include "arcane/core/CaseOptionSimple.h"
#include "arcane/core/CaseOptionEnum.h"
#include "arcane/core/CaseOptionExtended.h"
#include "arcane/core/CaseOptionComplexValue.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
namespace AxlOptionsBuilder
{
class Document;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe de base d'une liste d'options du jeu de données.
 *
 * Les instances de cette classe doivent toutes être allouées par
 * l'opérateur new() et ne doivent pas être détruite, le gestionnaire
 * de cas (ICaseMng) s'en chargeant.
 */
class ARCANE_CORE_EXPORT CaseOptions
: private ReferenceCounterImpl
, public ICaseOptions
{
  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();

 private:

  struct XmlContent
  {
    String m_xml_content;
    IXmlDocumentHolder* m_document = nullptr;
  };

 public:

  //! Construit un jeu d'options.
  CaseOptions(ICaseMng* cm, const String& name);
  //! Construit un jeu d'options.
  CaseOptions(ICaseOptionList*, const String& name);
  //! Construit un jeu d'options.
  CaseOptions(ICaseMng* cm, const String& name, const XmlNode& parent);
  //! Construit un jeu d'options.
  CaseOptions(ICaseOptionList*, const String& name, const XmlNode& parent, bool is_optional = false, bool is_multi = false);
  //! Construit un jeu d'options.

 protected:

  CaseOptions(ICaseMng*, const String& name, ICaseOptionList* parent);
  //! Construit un jeu d'options.
  CaseOptions(ICaseOptionList*, const String& name, ICaseOptionList* parent);

 private:

  friend class ServiceBuilderWithOptionsBase;

  // Uniquement pour ServiceBuilderWithOptionsBase
  static ReferenceCounter<ICaseOptions> createDynamic(ICaseMng* cm, const AxlOptionsBuilder::Document& options_doc);

  //! \internal
  CaseOptions(ICaseMng*, const XmlContent& xm_content);

 public:

  //! Libère les ressources
  ~CaseOptions() override;

 private:

  CaseOptions(const CaseOptions& rhs) = delete;
  CaseOptions& operator=(const CaseOptions& rhs) = delete;

 public:

  //! Retourne le vrai nom (non traduit) de l'élément.
  String rootTagTrueName() const override;

  //! Retourne le nom de l'élément dans le langage du jeu de données.
  String rootTagName() const override;

  //! Nom dans la langue \a lang de l'option. Retourne \a rootTagTrueName() si pas de traduction.
  String translatedName(const String& lang) const override;

  //! Retourne le vrai nom (non traduit) de l'élément.
  virtual String trueName() const { return rootTagTrueName(); }

  //! Retourne le nom de l'élément dans le langage du jeu de données.
  virtual String name() const { return rootTagName(); }

  void read(bool is_phase1) override
  {
    auto p = (is_phase1) ? eCaseOptionReadPhase::Phase1 : eCaseOptionReadPhase::Phase2;
    read(p);
  }

  void read(eCaseOptionReadPhase phase) override;

  void addInvalidChildren(XmlNodeList&) override;

  void printChildren(const String& lang, int indent) override;

  //! Retourne le service associé ou 0 s'il n'y en a pas.
  IServiceInfo* caseServiceInfo() const override;

  //! Retourne le module associé ou 0 s'il n'y en a pas.
  IModule* caseModule() const override;

  //! Associe le service \a m à ce jeu de données.
  void setCaseServiceInfo(IServiceInfo* m) override;

  //! Associe le module \a m à ce jeu de données.
  void setCaseModule(IModule* m) override;

  //! Ajoute à la liste \a col tous les options filles.
  void deepGetChildren(Array<CaseOptionBase*>& col) override;

  ICaseOptionList* configList() override;

  const ICaseOptionList* configList() const override;

  //! Fonction indiquant l'état d'activation de l'option
  ICaseFunction* activateFunction() override;

  /*!
    \brief Vrai si l'option est présente dans le fichier,
    faux s'il s'agit de la valeur par défaut.
  */
  bool isPresent() const override;

  /*!
    \brief Ajoute une traduction pour le nom de l'option.
    Ajoute le nom \a name de l'option correspondant au langage \a lang.
    Si une traduction existe déjà pour ce langage, elle est remplacée par
    celle-ci.
  */
  void addAlternativeNodeName(const String& lang, const String& name) override;

  ICaseMng* caseMng() const override;
  ITraceMng* traceMng() const override;
  ISubDomain* subDomain() const override;
  IMesh* mesh() const override;
  MeshHandle meshHandle() const override;
  ICaseDocument* caseDocument() const override;
  ICaseDocumentFragment* caseDocumentFragment() const override;

  void detach() override;

  void visit(ICaseDocumentVisitor* visitor) const override;

  String xpathFullName() const override;

  Ref<ICaseOptions> toReference() override;

 protected:

  friend class CaseOptionMultiServiceImpl;

  void _setTranslatedName();
  bool _setMeshHandleAndCheckDisabled(const String& mesh_name);

 protected:

  CaseOptionsPrivate* m_p; //!< Implémentation

 private:

  void _setMeshHandle(const MeshHandle& handle);
  void _setParent(ICaseOptionList* parent);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptionServiceImpl.h                                     (C) 2000-2025 */
/*                                                                           */
/* Implémentation d'une option du jeu de données utilisant un service.       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_CASEOPTIONSERVICEIMPL_H
#define ARCANE_CORE_CASEOPTIONSERVICEIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Functor.h"

#include "arcane/core/CaseOptions.h"
#include "arcane/core/CaseOptionsMulti.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IService;
class CaseOptionBuildInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un conteneur d'instances de service.
 * \todo: ajouter compteur de référence
 */
class ARCANE_CORE_EXPORT ICaseOptionServiceContainer
{
 public:
  virtual ~ICaseOptionServiceContainer() = default;
 public:
  virtual bool tryCreateService(Integer index,Internal::IServiceFactory2* factory,const ServiceBuildInfoBase& sbi) =0;
  virtual bool hasInterfaceImplemented(Internal::IServiceFactory2*) const =0;
  //! Alloue un tableau pour \a size éléments
  virtual void allocate(Integer size) =0;
  //! Retourne le nombre d'éléments du tableau.
  virtual Integer nbElem() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \ingroup CaseOption
 * \brief Classe de base de l'implémentation des options utilisant des services.
 *
 * Cette classe est interne à Arcane. La classe à utiliser est 'CaseOptionService'.
 */
class ARCANE_CORE_EXPORT CaseOptionServiceImpl
: public CaseOptions
{
 public:

  CaseOptionServiceImpl(const CaseOptionBuildInfo& cob,bool allow_null,bool is_optional);

 public:

  void read(eCaseOptionReadPhase phase) override;
  String serviceName() const { return m_service_name; }
  bool isOptional() const { return m_is_optional; }

  //! Retourne dans \a names les noms d'implémentations valides pour ce service
  virtual void getAvailableNames(StringArray& names) const;
  void visit(ICaseDocumentVisitor* visitor) const override;

  void setDefaultValue(const String& def_value);
  void addDefaultValue(const String& category,const String& value);

  /*!
   * \brief Positionne le conteneur d'instances.
   *
   * \a container reste la propriété de l'appelant qui doit gérer
   * sa durée de vie.
   */
  void setContainer(ICaseOptionServiceContainer* container);

  void setMeshName(const String& mesh_name) { m_mesh_name = mesh_name; }
  String meshName() const { return m_mesh_name; }

 protected:

  virtual void print(const String& lang,std::ostream& o) const;

 protected:

  String _defaultValue() const { return m_default_value; }

 private:

  String m_name;
  String m_default_value;
  String m_service_name;
  String m_mesh_name;
  XmlNode m_element; //!< Element de l'option
  bool m_allow_null;
  bool m_is_optional;
  bool m_is_override_default;
  //! Liste des valeurs par défaut par catégorie.
  StringDictionary m_default_values;
  ICaseOptionServiceContainer* m_container;

 private:

  void _readPhase1();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe de base d'une option service pouvant être présente plusieurs fois.
 *
 * Il faut appeler setContainer() pour positionner un conteneur
 * avant d'utiliser cette classe.
 */
class ARCANE_CORE_EXPORT CaseOptionMultiServiceImpl
: public CaseOptionsMulti
{
 public:

  CaseOptionMultiServiceImpl(const CaseOptionBuildInfo& cob,bool allow_null);
  ~CaseOptionMultiServiceImpl();

 public:

  //! Retourne dans \a names les noms d'implémentations valides pour ce service
  void getAvailableNames(StringArray& names) const;
  //! Nom du n-ième service
  String serviceName(Integer index) const
  {
    return m_services_name[index];
  }

  void multiAllocate(const XmlNodeList&) override;
  void visit(ICaseDocumentVisitor* visitor) const override;
  /*!
   * \brief Positionne le conteneur d'instances.
   *
   * \a container reste la propriété de l'appelant qui doit gérer
   * sa durée de vie.
   */
  void setContainer(ICaseOptionServiceContainer* container);

  void setMeshName(const String& mesh_name) { m_mesh_name = mesh_name; }
  String meshName() const { return m_mesh_name; }

 public:

  void _setNotifyAllocateFunctor(IFunctor* f)
  {
    m_notify_functor = f;
  }

 protected:

  String _defaultValue() const { return m_default_value; }

 protected:

  bool m_allow_null;
  String m_default_value;
  String m_mesh_name;
  IFunctor* m_notify_functor;
  ICaseOptionServiceContainer* m_container;
  //! Liste des options allouées qu'il faudra supprimer.
  UniqueArray<ReferenceCounter<ICaseOptions>> m_allocated_options;
  //! Noms du service pour chaque occurence
  UniqueArray<String> m_services_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

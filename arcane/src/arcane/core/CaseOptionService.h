// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptionService.h                                         (C) 2000-2025 */
/*                                                                           */
/* Options du jeu de données utilisant un service.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_CASEOPTIONSERVICE_H
#define ARCANE_CORE_CASEOPTIONSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Functor.h"

#include "arcane/core/CaseOptions.h"
#include "arcane/core/IServiceMng.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/ServiceUtils.h"
#include "arcane/core/ArcaneException.h"
#include "arcane/core/IFactoryService.h"
#include "arcane/core/IServiceFactory.h"
#include "arcane/core/StringDictionary.h"
#include "arcane/core/CaseOptionServiceImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IService;
class CaseOptionBuildInfo;
template<typename T> class CaseOptionServiceT;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation du conteneur pour un service de type \a InterfaceType.
 */
template<typename InterfaceType>
class CaseOptionServiceContainer
: public ICaseOptionServiceContainer
{
 public:
  ~CaseOptionServiceContainer() override
  {
    removeInstances();
  }

  bool tryCreateService(Integer index,Internal::IServiceFactory2* factory,const ServiceBuildInfoBase& sbi) override
  {
    auto true_factory = dynamic_cast< Internal::IServiceFactory2T<InterfaceType>* >(factory);
    if (true_factory){
      Ref<InterfaceType> sr = true_factory->createServiceReference(sbi);
      InterfaceType* s = sr.get();
      m_services_reference[index] = sr;
      m_services[index] = s;
      return (s != nullptr);
    }
    return false;
  }

  bool hasInterfaceImplemented(Internal::IServiceFactory2* factory) const override
  {
    auto true_factory = dynamic_cast< Internal::IServiceFactory2T<InterfaceType>* >(factory);
    if (true_factory){
      return true;
    }
    return false;
  }

  //! Alloue un tableau pour \a size éléments
  void allocate(Integer asize) override
  {
    m_services.resize(asize,nullptr);
    m_services_reference.resize(asize);
  }

  //! Retourne le nombre d'éléments du tableau.
  Integer nbElem() const override
  {
    return m_services.size();
  }

  InterfaceType* child(Integer i) const
  {
    return m_services[i];
  }

  Ref<InterfaceType> childRef(Integer i) const
  {
    return m_services_reference[i];
  }

 public:
  //! Supprime les instances des services
  void removeInstances()
  {
    m_services_reference.clear();
    m_services.clear();
  }
 public:
  ArrayView<InterfaceType*> view() { return m_services; }
 private:
  UniqueArray<InterfaceType*> m_services;
  UniqueArray<Ref<InterfaceType>> m_services_reference;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup CaseOption
 * \brief Classe de base des options utilisant des services.
 *
 * Les instances de cette classe ne sont pas copiables.
 */
class ARCANE_CORE_EXPORT CaseOptionService
{
 public:

 CaseOptionService(const CaseOptionBuildInfo& cob,bool allow_null,bool is_optional)
  : m_impl(new CaseOptionServiceImpl(cob,allow_null,is_optional))
  {
  }

  virtual ~CaseOptionService() = default;

 public:

  CaseOptionService(const CaseOptionService&) = delete;
  const CaseOptionService& operator=(const CaseOptionService&) = delete;

 public:

  ARCANE_DEPRECATED_REASON("Y2022: Use toICaseOptions() instead")
  operator CaseOptions& () { return *_impl(); }

  ARCANE_DEPRECATED_REASON("Y2022: Use toICaseOptions() instead")
  operator const CaseOptions& () const { return *_impl(); }

 public:

   const ICaseOptions* toICaseOptions() { return _impl(); }

 public:

  String rootTagName() const { return m_impl->rootTagName(); }
  String name() const { return m_impl->name(); }
  String serviceName() const { return m_impl->serviceName(); }
  bool isOptional() const { return m_impl->isOptional(); }
  bool isPresent() const { return m_impl->isPresent(); }
  void addAlternativeNodeName(const String& lang,const String& name)
  {
    m_impl->addAlternativeNodeName(lang,name);
  }
  void getAvailableNames(StringArray& names) const
  {
    m_impl->getAvailableNames(names);
  }
  /*!
   * \brief Positionne la valeur par défaut du nom du service.
   *
   * Si l'option n'est pas pas présente dans le jeu de données, alors sa valeur sera
   * celle spécifiée par l'argument \a def_value, sinon l'appel de cette méthode est sans effet.
   *
   * Cette méthode ne peut être apellée que lors de la phase 1 de la lecture
   * du jeu de données car par la suite le service est déjà instancié. Une exception
   * FatalErrorException est levé si cette méthode est appelée et que le service
   * est déjà instancié.
   */
  void setDefaultValue(const String& def_value)
  {
    m_impl->setDefaultValue(def_value);
  }
  //! Ajoute la valeur par défaut \a value à la catégorie \a category
  void addDefaultValue(const String& category,const String& value)
  {
    m_impl->addDefaultValue(category,value);
  }
  /*!
   * \brief Positionne le nom du maillage auquel le service sera associé.
   *
   * Si nul, le service est associé au maillage par défaut du sous-domaine
   * (ISubDomain::defaultMeshHandle()). L'association réelle se fait lors de la
   * lecture des options. Appeler cette méthode après lecture des options n'aura
   * aucun impact.
   */
  void setMeshName(const String& mesh_name);

  /*!
   * \brief Nom du maillage auquel le service est associé.
   *
   * Il s'agit du nom du maillage tel que spécifié dans le descripteur de service
   * (le fichier 'axl'). Pour obtenir le maillage associé après lecture des options
   * il faut utiliser ICaseOptions::meshHandle().
   */
  String meshName() const;

 protected:

  CaseOptionServiceImpl* _impl() { return m_impl.get(); }
  const CaseOptionServiceImpl* _impl() const { return m_impl.get(); }

 private:

  ReferenceCounter<CaseOptionServiceImpl> m_impl;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class InterfaceType>
class CaseOptionServiceT
: public CaseOptionService
{
 public:
  CaseOptionServiceT(const CaseOptionBuildInfo& cob,bool allow_null,bool is_optional)
  : CaseOptionService(cob,allow_null,is_optional)
  {
    _impl()->setContainer(&m_container);
  }
  ~CaseOptionServiceT() = default;
 public:
  InterfaceType* operator()() const { return _instance(); }
  InterfaceType* instance() const { return _instance(); }
  Ref<InterfaceType> instanceRef() const { return _instanceRef(); }
 private:
  CaseOptionServiceContainer<InterfaceType> m_container;
 private:
  InterfaceType* _instance() const
  {
    if (m_container.nbElem()==1)
      return m_container.child(0);
    return nullptr;
  }
  Ref<InterfaceType> _instanceRef() const
  {
    if (m_container.nbElem()==1)
      return m_container.childRef(0);
    return {};
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup CaseOption
 * \brief Classe de base d'une option service pouvant être présente plusieurs fois.
 */
class ARCANE_CORE_EXPORT CaseOptionMultiService
{
 public:
  CaseOptionMultiService(const CaseOptionBuildInfo& cob,bool allow_null)
  : m_impl(new CaseOptionMultiServiceImpl(cob,allow_null))
  {
  }
  virtual ~CaseOptionMultiService() = default;
  CaseOptionMultiService(const CaseOptionMultiService&) = delete;
  const CaseOptionMultiService& operator=(const CaseOptionMultiService&) = delete;
 public:
  XmlNode rootElement() const { return m_impl->toCaseOptions()->configList()->rootElement(); }
  String rootTagName() const { return m_impl->rootTagName(); }
  String name() const { return m_impl->name(); }
  //! Retourne dans \a names les noms d'implémentations valides pour ce service
  void getAvailableNames(StringArray& names) const
  {
    m_impl->getAvailableNames(names);
  }
  //! Nom du n-ième service
  String serviceName(Integer index) const 
  {
    return m_impl->serviceName(index);
  }
  void addAlternativeNodeName(const String& lang,const String& name)
  {
    m_impl->addAlternativeNodeName(lang,name);
  }
  /*!
   * \brief Positionne le nom du maillage auquel le service sera associé.
   *
   * \sa CaseOptionService::setMeshName()
   */
  void setMeshName(const String& mesh_name);

  /*!
   * \brief Nom du maillage auquel le service est associé.
   *
   * \sa CaseOptionService::axlMeshName();
   */
  String meshName() const;

 protected:

  CaseOptionMultiServiceImpl* _impl() { return m_impl.get(); }
  const CaseOptionMultiServiceImpl* _impl() const { return m_impl.get(); }

 private:

  ReferenceCounter<CaseOptionMultiServiceImpl> m_impl;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup CaseOption
 * \brief Option du jeu de données de type liste de services.
 */
template<typename InterfaceType>
class CaseOptionMultiServiceT
: public CaseOptionMultiService
, public ArrayView<InterfaceType*>
{
  typedef CaseOptionMultiServiceT<InterfaceType> ThatClass;
 public:
  CaseOptionMultiServiceT(const CaseOptionBuildInfo& cob,bool allow_null)
  : CaseOptionMultiService(cob,allow_null)
  , m_notify_functor(this,&ThatClass::_notify)
  {
    _impl()->setContainer(&m_container);
    _impl()->_setNotifyAllocateFunctor(&m_notify_functor);
  }
 public:
  CaseOptionMultiServiceT<InterfaceType>& operator()()
  {
    return *this;
  }
  const CaseOptionMultiServiceT<InterfaceType>& operator()() const
  {
    return *this;
  }
 protected:
  // Notification par l'implémentation
  void _notify()
  {
    this->setArray(m_container.view());
  }
 private:
  CaseOptionServiceContainer<InterfaceType> m_container;
  FunctorT<ThatClass> m_notify_functor;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

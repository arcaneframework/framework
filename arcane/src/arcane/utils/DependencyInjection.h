// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DependencyInjection.h                                       (C) 2000-2021 */
/*                                                                           */
/* Types et fonctions pour gérer le pattern 'DependencyInjection'.           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_DEPENDENCYINJECTION_H
#define ARCANE_DEPENDENCYINJECTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * NOTE: API en cours de définition. Ne pas utiliser.
 */

#include "arcane/utils/Ref.h"
#include "arcane/utils/ExternalRef.h"

#include <functional>
#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::DependencyInjection
{

class Injector;
class FactoryInfo;
namespace impl
{
class IInstanceFactory;
template<typename InterfaceType>
class IConcreteFactory;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_UTILS_EXPORT IInjectedInstance
{
 public:

  virtual ~IInjectedInstance() = default;
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_UTILS_EXPORT ProviderProperty
{
 public:
  ProviderProperty(const char* name) : m_name(name){}
 public:
  const char* m_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_UTILS_EXPORT FactoryInfo
{
 private:
  FactoryInfo(const ProviderProperty& property) : m_property(property){}
 public:
  static FactoryInfo* create(const ProviderProperty& property,const char* file_name,int line_number)
  {
    ARCANE_UNUSED(file_name);
    ARCANE_UNUSED(line_number);
    return new FactoryInfo(property);
  }
  void addFactory(impl::IInstanceFactory* f){}
 private:
  ProviderProperty m_property;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface typée gérant l'instance d'un service.
 */
template<typename InterfaceType>
class IInjectedInstanceT
: public IInjectedInstance
{
 public:
  virtual Ref<InterfaceType> instance() =0;
 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface typée gérant l'instance d'un service.
 */
template<typename InterfaceType>
class InjectedInstance
: public IInjectedInstanceT<InterfaceType>
{
 public:
  InjectedInstance(Ref<InterfaceType> t_instance) : m_instance(t_instance){}
 public:
  Ref<InterfaceType> instance() override { return m_instance; }
 private:
  Ref<InterfaceType> m_instance;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Référence sur une instance injectée.
 *
 * Cette classe est gérée via un compteur de référence à la manière
 * de la classe std::shared_ptr.
 */
class ARCANE_UTILS_EXPORT InjectedInstanceRef
{
  typedef Ref<IInjectedInstance> RefType;
 private:
  InjectedInstanceRef(const RefType& r) : m_instance(r){}
 public:
  InjectedInstanceRef() = default;
 public:
  static InjectedInstanceRef createRef(IInjectedInstance* p)
  {
    return InjectedInstanceRef(RefType::create(p));
  }
  static InjectedInstanceRef createRefNoDestroy(IInjectedInstance* p)
  {
    return InjectedInstanceRef(RefType::_createNoDestroy(p));
  }
  static InjectedInstanceRef createWithHandle(IInjectedInstance* p,Internal::ExternalRef handle)
  {
    return InjectedInstanceRef(RefType::createWithHandle(p,handle));
  }
 public:
  IInjectedInstance* get() const { return m_instance.get(); }
  void reset() { m_instance.reset(); }
 private:
  RefType m_instance;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 */
class ARCANE_UTILS_EXPORT IInstanceFactory
{
 protected:

  virtual ~IInstanceFactory() = default;

 public:

  virtual InjectedInstanceRef createReference(Injector&) =0;

  virtual const FactoryInfo* factoryInfo() const =0;

 public:

  //! Ajoute une référence.
  virtual void addReference() =0;

  //! Supprime une référence.
  virtual void removeReference() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief internal.
 * \brief Classe de base pour une fabrique pour un service.
 *
 * Cette classe s'utiliser via un ReferenceCounter pour gérer sa destruction.
 */
class ARCANE_CORE_EXPORT AbstractInstanceFactory
: public IInstanceFactory
{
 protected:
  AbstractInstanceFactory() : m_nb_ref(0){}
 public:
  void addReference() override { ++m_nb_ref; }
  void removeReference() override
  {
    Int32 v = std::atomic_fetch_add(&m_nb_ref,-1);
    if (v==1)
      delete this;
  }
 private:
  std::atomic<Int32> m_nb_ref;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename InterfaceType>
class InstanceFactory
: public AbstractInstanceFactory
{
 public:

  InstanceFactory(FactoryInfo* si,IConcreteFactory<InterfaceType>* sub_factory)
  : m_factory_info(si), m_sub_factory(sub_factory)
  {
  }

  ~InstanceFactory() override
  {
    delete m_sub_factory;
  }

  InjectedInstanceRef createReference(Injector& injector) override
  {
    return _create(_createReference(injector));
  }

  const FactoryInfo* factoryInfo() const override
  {
    return m_factory_info;
  }

 protected:

  FactoryInfo* m_factory_info;
  IConcreteFactory<InterfaceType>* m_sub_factory;

 private:

  Ref<InterfaceType> _createReference(Injector& injector)
  {
    return m_sub_factory->createReference(injector);
  }

  InjectedInstanceRef _create(Ref<InterfaceType> it)
  {
    IInjectedInstance* x = (!it) ? nullptr : new InjectedInstance<InterfaceType>(it);
    return InjectedInstanceRef::createRef(x);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'un fonctor de création d'une instance de service
 * correspondant à l'interface \a InterfaceType.
 */
template<typename InterfaceType>
class IConcreteFactory
{
 public:
  virtual ~IConcreteFactory() = default;
 public:
  //! Créé une instance du service .
  virtual Ref<InterfaceType> createReference(Injector&) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Fabrique pour le service \a ServiceType pour l'interface \a InterfaceType.
 */
template<typename ServiceType,typename InterfaceType>
class ConcreteFactory
: public IConcreteFactory<InterfaceType>
{
 public:

  Ref<InterfaceType> createReference(Injector& injector) override
  {
    ServiceType* st = new ServiceType(injector);
    return makeRef<InterfaceType>(st);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 */
class ARCANE_UTILS_EXPORT Injector
{
  class Impl;

 public:

  Injector();
  Injector(const Injector&) = delete;
  Injector& operator=(const Injector&) = delete;
  ~Injector() = default;

 public:

  template<typename InterfaceType> void
  bind(Ref<InterfaceType> iref)
  {
    auto* x = new impl::InjectedInstance<InterfaceType>(iref);
    _add(x);
  }

  template<typename InterfaceType> Ref<InterfaceType>
  get()
  {
    using InjectedInstanceType = impl::IInjectedInstanceT<InterfaceType>;
    InjectedInstanceType* t = nullptr;
    auto f = [&](IInjectedInstance* v) -> bool
             {
               t = dynamic_cast<InjectedInstanceType*>(v);
               return t;
             };
    _iterate(f);
    if (t)
      return t->instance();
    // TODO: faire un fatal ou créér l'instance
    return {};
  }

 private:

  Impl* m_p = nullptr;

 private:

  void _add(IInjectedInstance* instance);
  // Itère sur la lambda et s'arrête dès que cette dernière retourne \a true
  template<typename Lambda> void
  _iterate(const Lambda& lambda)
  {
    Integer n = _nbValue();
    for (Integer i=0; i<n; ++i ){
      if (lambda(_value(i)))
        return;
    }
  }
  Integer _nbValue() const;
  IInjectedInstance* _value(Integer i) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_UTILS_EXPORT GlobalRegisterer
{
 public:

  typedef FactoryInfo* (*FactoryCreateFunc)(const ProviderProperty& property);

 public:

  /*!
   * \brief Crée en enregistreur pour le service \a name et la fonction \a func.
   *
   * Ce constructeur est utilisé pour enregistrer un service.
   */
 GlobalRegisterer(FactoryCreateFunc func,const ProviderProperty& property) ARCANE_NOEXCEPT;

 public:

  FactoryCreateFunc infoCreatorWithPropertyFunction() { return m_factory_create_func; }

  //! Nom du service
  const char* name() { return m_name; }

  const ProviderProperty& property() const { return m_factory_property; }

  //! Service précédent (0 si le premier)
  GlobalRegisterer* previousService() const { return m_previous; }

  //! Service suivant (0 si le dernier)
  GlobalRegisterer* nextService() const { return m_next; }

 public:

  //! Accès au premier élément de la chaine d'enregistreur de service
  static GlobalRegisterer* firstService();

  //! Nombre d'enregisteur de service dans la chaine
  static Integer nbService();

 private:

  FactoryCreateFunc m_factory_create_func = nullptr;
  const char* m_name = nullptr;
  ProviderProperty m_factory_property;
  GlobalRegisterer* m_previous = nullptr;
  GlobalRegisterer* m_next = nullptr;

 private:

  void _init();

  //! Positionne le service précédent
  /*! Utilisé en interne pour construire la chaine de service */
  void _setPreviousService(GlobalRegisterer* s) { m_previous = s; }

  //! Positionne le service suivant
  /*! Utilisé en interne pour construire la chaine de service */
  void _setNextService(GlobalRegisterer* s) { m_next = s; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe permettant d'enregistrer une fabrique pour un service
 * implémentant l'interface \a InterfaceType.
 */
template<typename InterfaceType>
class ServiceInterfaceRegisterer
{
 public:

  typedef InterfaceType Interface;

  explicit ServiceInterfaceRegisterer(const char* name)
  : m_name(name), m_namespace_name(nullptr)
  {
  }

  ServiceInterfaceRegisterer(const char* namespace_name,const char* name)
  : m_name(name), m_namespace_name(namespace_name)
  {
  }

 public:

  //! Enregistre dans \a si une fabrique pour créer une instance du service \a ServiceType
  template<typename ServiceType> void
  registerToFactoryInfo(FactoryInfo* si) const
  {
    impl::IConcreteFactory<InterfaceType>* factory = new impl::ConcreteFactory<ServiceType,InterfaceType>();
#if 0
    if (m_namespace_name)
      si->addImplementedInterface(String(m_namespace_name)+String("::")+String(m_name));
    else
      si->addImplementedInterface(m_name);
    //si->addFactory(new ServiceFactory2TV2<InterfaceType>(si,factory));
#endif
    si->addFactory(new impl::InstanceFactory<InterfaceType>(si,factory));
  }

 private:

  const char* m_name;
  const char* m_namespace_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe permettant de créer et d'enregistrer les fabriques pour un service.
 */
template<typename ServiceType>
class ServiceAllInterfaceRegisterer
{
 private:

  //! Surcharge pour 1 interface
  template<typename InterfaceType> static void
  _create(FactoryInfo* si,const InterfaceType& i1)
  {
    i1.template registerToFactoryInfo<ServiceType>(si);
  }
  //! Surcharge pour 2 interfaces ou plus
  template<typename I1,typename I2,typename ... OtherInterfaces>
  static void _create(FactoryInfo* si,const I1& i1,const I2& i2,const OtherInterfaces& ... args)
  {
    _create<I1>(si,i1);
    // Applique la récursivité sur les types restants
    _create<I2,OtherInterfaces...>(si,i2,args...);
  }

 public:

  //! Enregistre dans le service les fabriques pour les interfacs \a Interfaces
  template<typename ... Interfaces> static void
  registerProviderInfo(FactoryInfo* si, const Interfaces& ... args)
  {
    //si->setSingletonFactory(new Internal::SingletonServiceFactory<ServiceType,typename Interfaces::Interface ... >(si));
    _create(si,args...);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCANE_DI_SERVICE_INTERFACE(ainterface)\
  ::Arcane::DependencyInjection::ServiceInterfaceRegisterer< ainterface >(#ainterface)

#define ARCANE_DI_SERVICE_INTERFACE_NS(ainterface_ns,ainterface) \
  ::Arcane::DependencyInjection::ServiceInterfaceRegisterer<ainterface_ns :: ainterface>(#ainterface_ns,#ainterface)

// TODO: garantir au moins une interface

#define ARCANE_DI_REGISTER_PROVIDER(t_class,t_provider_property,...)  \
namespace\
{\
  Arcane::DependencyInjection::FactoryInfo*                            \
  ARCANE_JOIN_WITH_LINE(arcaneCreateDependencyInjectionProviderInfo##t_class) (const Arcane::DependencyInjection::ProviderProperty& property) \
  {\
    auto* si = Arcane::DependencyInjection::FactoryInfo::create(property,__FILE__,__LINE__); \
    Arcane::DependencyInjection::ServiceAllInterfaceRegisterer<t_class> :: registerProviderInfo(si,__VA_ARGS__); \
    return si;\
  }\
}\
Arcane::DependencyInjection::GlobalRegisterer ARCANE_EXPORT ARCANE_JOIN_WITH_LINE(globalServiceRegisterer##aclass) \
  (& ARCANE_JOIN_WITH_LINE(arcaneCreateDependencyInjectionProviderInfo##t_class),t_provider_property)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::DependencyInjection

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

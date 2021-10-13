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
#ifndef ARCANE_UTILS_DEPENDENCYINJECTION_H
#define ARCANE_UTILS_DEPENDENCYINJECTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * NOTE: API en cours de définition. Ne pas utiliser.
 */

#include "arcane/utils/Ref.h"
#include "arcane/utils/ExternalRef.h"

#include "arccore/base/ReferenceCounterImpl.h"

//TODO Mettre le lancement des exceptions dans le '.cc'
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/FatalErrorException.h"

#include <tuple>
#include <typeinfo>

// TODO: Améliorer les messages d'erreurs en cas d'échec de l'injection
// TOOD: Mettre les méthodes qui lancent les exception dans 'DependencyInjection.cc'
// TODO: Ajouter méthodes pour afficher le type en cas d'erreur (utiliser A_FUNC_INFO)
// TODO: Ajouter mode verbose
// TODO: Supporter plusieurs constructeurs et ne pas échouer si le premier
// ne fonctionne pas
// TODO: Supporter des instances externes (comme par exemple celles créées en C#).

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::DependencyInjection
{
class Injector;
}

namespace Arcane::DependencyInjection::impl
{
class FactoryInfo;
class IInstanceFactory;
template <typename InterfaceType>
class IConcreteFactory;
class ConcreteFactoryTypeInfo;
} // namespace Arcane::DependencyInjection::impl

ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS(Arcane::DependencyInjection::impl::IInstanceFactory)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::DependencyInjection
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_UTILS_EXPORT IInjectedInstance
{
 public:
  virtual ~IInjectedInstance() = default;
  virtual bool hasName(const String& str) const = 0;
  virtual bool hasTypeInfo(const std::type_info& tinfo) const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \warning Cette classe est utilisées dans des constructeurs/destructeurs
 * globaux et ne doit pas faire d'allocation/désallocation ni utiliser de
 * types qui en font.
 */
class ARCANE_UTILS_EXPORT ProviderProperty
{
 public:
  ProviderProperty(const char* name)
  : m_name(name)
  {}

 public:
  const char* name() const { return m_name; }

 private:
  const char* m_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::DependencyInjection

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::DependencyInjection::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TypeInfo
{
 private:
  TypeInfo(const TraceInfo& trace_info, const std::type_info& type_info)
  : m_trace_info(trace_info)
  , m_type_info(type_info)
  {}

 public:
  template <typename Type> static TypeInfo create()
  {
    return TypeInfo(A_FUNCINFO, typeid(Type));
  }
  const TraceInfo& traceInfo() const { return m_trace_info; }
  const std::type_info& stdTypeInfo() const { return m_type_info; }

 private:
  TraceInfo m_trace_info;
  const std::type_info& m_type_info;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_UTILS_EXPORT ConcreteFactoryTypeInfo
{
 private:
  ConcreteFactoryTypeInfo(TypeInfo&& a, TypeInfo&& b, TypeInfo&& c)
  : m_interface_info(a)
  , m_concrete_info(b)
  , m_constructor_info(c)
  {}

 public:
  template <typename InterfaceType, typename ConcreteType, typename ConstructorType>
  static ConcreteFactoryTypeInfo create()
  {
    return ConcreteFactoryTypeInfo(TypeInfo::create<InterfaceType>(),
                                   TypeInfo::create<ConcreteType>(),
                                   TypeInfo::create<ConstructorType>());
  }
  const TypeInfo& interfaceTypeInfo() const { return m_interface_info; }
  const TypeInfo& concreteTypeInfo() const { return m_concrete_info; }
  const TypeInfo& constructorTypeInfo() const { return m_constructor_info; }

 private:
  TypeInfo m_interface_info;
  TypeInfo m_concrete_info;
  TypeInfo m_constructor_info;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface typée gérant l'instance d'un service.
 */
template <typename InterfaceType>
class IInjectedRefInstanceT
: public IInjectedInstance
{
 public:
  virtual Ref<InterfaceType> instance() = 0;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface typée gérant l'instance d'un service.
 */
template <typename InterfaceType>
class InjectedRefInstance
: public IInjectedRefInstanceT<InterfaceType>
{
 public:
  using InstanceType = Ref<InterfaceType>;

 public:
  InjectedRefInstance(InstanceType t_instance, const String& t_name)
  : m_instance(t_instance)
  , m_name(t_name)
  {}

 public:
  Ref<InterfaceType> instance() override { return m_instance; }
  bool hasName(const String& str) const override { return m_name == str; }
  bool hasTypeInfo(const std::type_info& tinfo) const override { return typeid(InstanceType) == tinfo; }

 private:
  InstanceType m_instance;
  String m_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface typée gérant une instance
 */
template <typename Type>
class IInjectedValueInstance
: public IInjectedInstance
{
 public:
  virtual Type instance() const = 0;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface typée gérant l'instance d'un service.
 */
template <typename Type>
class InjectedValueInstance
: public IInjectedValueInstance<Type>
{
 public:
  using InstanceType = Type;

 public:
  InjectedValueInstance(Type t_instance, const String& t_name)
  : m_instance(t_instance)
  , m_name(t_name)
  {}

 public:
  Type instance() const override { return m_instance; }
  bool hasName(const String& str) const override { return m_name == str; }
  bool hasTypeInfo(const std::type_info& tinfo) const override { return typeid(InstanceType) == tinfo; }

 private:
  Type m_instance;
  String m_name;
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
  InjectedInstanceRef(const RefType& r)
  : m_instance(r)
  {}

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
  static InjectedInstanceRef createWithHandle(IInjectedInstance* p, Internal::ExternalRef handle)
  {
    return InjectedInstanceRef(RefType::createWithHandle(p, handle));
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
 * \brief Fabrique pour une instance encapsulée par une référence (i.e Ref<T>).
 */
class ARCANE_UTILS_EXPORT IInstanceFactory
{
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();

 protected:
  virtual ~IInstanceFactory() = default;

 public:
  virtual InjectedInstanceRef createGenericReference(Injector& injector, const String& name) = 0;

  virtual const FactoryInfo* factoryInfo() const = 0;

  virtual ConcreteFactoryTypeInfo concreteFactoryInfo() const = 0;

  virtual Int32 nbConstructorArg() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief internal.
 * \brief Classe de base pour une fabrique.
 *
 * Cette classe s'utiliser via un ReferenceCounter pour gérer sa destruction.
 */
class ARCANE_UTILS_EXPORT AbstractInstanceFactory
: public ReferenceCounterImpl
, public IInstanceFactory
{
  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename InterfaceType>
class InstanceFactory
: public AbstractInstanceFactory
{
 public:
  InstanceFactory(FactoryInfo* si, IConcreteFactory<InterfaceType>* sub_factory)
  : m_factory_info(si)
  , m_sub_factory(sub_factory)
  {
  }

  ~InstanceFactory() override
  {
    delete m_sub_factory;
  }

  InjectedInstanceRef createGenericReference(Injector& injector, const String& name) override
  {
    return _create(_createReference(injector), name);
  }

  Ref<InterfaceType> createReference(Injector& injector)
  {
    return _createReference(injector);
  }

  const FactoryInfo* factoryInfo() const override
  {
    return m_factory_info;
  }

  ConcreteFactoryTypeInfo concreteFactoryInfo() const override
  {
    return m_sub_factory->concreteFactoryInfo();
  }

  Int32 nbConstructorArg() const override
  {
    return m_sub_factory->nbConstructorArg();
  }

 protected:
  FactoryInfo* m_factory_info;
  IConcreteFactory<InterfaceType>* m_sub_factory;

 private:
  Ref<InterfaceType> _createReference(Injector& injector)
  {
    return m_sub_factory->createReference(injector);
  }

  InjectedInstanceRef _create(Ref<InterfaceType> it, const String& name)
  {
    IInjectedInstance* x = (!it) ? nullptr : new InjectedRefInstance<InterfaceType>(it, name);
    return InjectedInstanceRef::createRef(x);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_UTILS_EXPORT IConcreteFactoryBase
{
 public:
  virtual ~IConcreteFactoryBase() = default;

 public:
  virtual ConcreteFactoryTypeInfo concreteFactoryInfo() const = 0;
  virtual Int32 nbConstructorArg() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'un fonctor de création d'une instance de service
 * correspondant à l'interface \a InterfaceType.
 */
template <typename InterfaceType>
class IConcreteFactory
: public IConcreteFactoryBase
{
 public:
  virtual ~IConcreteFactory() = default;

 public:
  //! Créé une instance du service .
  virtual Ref<InterfaceType> createReference(Injector&) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Fabrique pour le service \a ServiceType pour l'interface \a InterfaceType.
 */
template <typename ServiceType, typename InterfaceType>
class ConcreteFactory
: public IConcreteFactory<InterfaceType>
{
 public:
  Ref<InterfaceType> createReference(Injector& injector) override
  {
    ServiceType* st = new ServiceType(injector);
    return makeRef<InterfaceType>(st);
  }
  ConcreteFactoryTypeInfo concreteFactoryInfo() const override
  {
    return ConcreteFactoryTypeInfo::create<InterfaceType, ServiceType, std::tuple<>>();
  }
  // Pour indiquer qu'on n'utilise pas le nombre d'arguments, on met (-1).
  Int32 nbConstructorArg() const override { return -1; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Informations pour une fabrique.
 */
class ARCANE_UTILS_EXPORT FactoryInfo
{
  friend class Arcane::DependencyInjection::Injector;
  class Impl;

 public:
  FactoryInfo(const FactoryInfo&) = delete;
  FactoryInfo& operator=(const FactoryInfo&) = delete;
  ~FactoryInfo();

 private:
  FactoryInfo(const ProviderProperty& property);

 public:
  static FactoryInfo* create(const ProviderProperty& property, const char* file_name, int line_number)
  {
    ARCANE_UNUSED(file_name);
    ARCANE_UNUSED(line_number);
    return new FactoryInfo(property);
  }
  void addFactory(Ref<IInstanceFactory> f);
  bool hasName(const String& str) const;

 private:
  Impl* m_p = nullptr;
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
  GlobalRegisterer(FactoryCreateFunc func, const ProviderProperty& property) ARCANE_NOEXCEPT;

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
template <typename InterfaceType>
class ServiceInterfaceRegisterer
{
 public:
  static constexpr bool isConstructor() { return false; }

  typedef InterfaceType Interface;

  explicit ServiceInterfaceRegisterer(const char* name)
  : m_name(name)
  , m_namespace_name(nullptr)
  {
  }

  ServiceInterfaceRegisterer(const char* namespace_name, const char* name)
  : m_name(name)
  , m_namespace_name(namespace_name)
  {
  }

 public:
  //! Enregistre dans \a si une fabrique pour créer une instance du service \a ServiceType
  template <typename ServiceType> void
  registerToFactoryInfo(FactoryInfo* si) const
  {
    IConcreteFactory<InterfaceType>* factory = new ConcreteFactory<ServiceType, InterfaceType>();
    si->addFactory(makeRef<IInstanceFactory>(new InstanceFactory<InterfaceType>(si, factory)));
  }

 private:
  const char* m_name;
  const char* m_namespace_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::DependencyInjection::impl

namespace Arcane::DependencyInjection
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 */
class ARCANE_UTILS_EXPORT Injector
{
  class Impl;

  template <class Type>
  class InjectorHelper
  {
   public:
    static IInjectedInstance* bind(const Type& t, const String& name)
    {
      return new impl::InjectedValueInstance<Type>(t, name);
    }
    static Type get(Injector& i, const String& name)
    {
      return i._getValue<Type>(name);
    }
  };

  //! Spécialisation pour les 'Ref'.
  template <class PointerType>
  class InjectorHelper<Ref<PointerType>>
  {
   public:
    using ThatType = Ref<PointerType>;

   public:
    static IInjectedInstance* bind(const ThatType& t, const String& name)
    {
      return new impl::InjectedRefInstance<PointerType>(t, name);
    }
    static ThatType get(Injector& i, const String& name)
    {
      return i._getRef<PointerType>(name);
    }
  };

 public:
  Injector();
  Injector(const Injector&) = delete;
  Injector& operator=(const Injector&) = delete;
  ~Injector() = default;

 public:
  template <typename Type> void
  bind(Type iref, const String& name = String())
  {
    _add(InjectorHelper<Type>::bind(iref, name));
  }

  template <typename Type> Type
  get(const String& name = String())
  {
    return InjectorHelper<Type>::get(*this, name);
  }

  template <typename InterfaceType> Ref<InterfaceType>
  createInstance(const String& service_name = String())
  {
    using FactoryType = impl::InstanceFactory<InterfaceType>;
    Ref<InterfaceType> instance;
    Integer nb_instance = _nbValue();
    //std::cout << "NB_INSTANCE=" << nb_instance << "\n";
    // Il faut trouver un constructeur qui ait le même nombre d'arguments que le nombre d'instances
    // enregistrées
    auto f = [&](impl::IInstanceFactory* v) -> bool {
      //std::cout << "TRY DYNAMIC_CAST FACTORY v=" << v << " n=" << v->nbConstructorArg() << "\n";
      Int32 nb_constructor_arg = v->nbConstructorArg();
      if (nb_constructor_arg >= 0 && nb_constructor_arg != nb_instance)
        return false;
      auto* t = dynamic_cast<FactoryType*>(v);
      //std::cout << "TRY DYNAMIC_CAST FACTORY v=" << v << " t=" << t << "\n";
      if (t) {
        Ref<InterfaceType> x = t->createReference(*this);
        if (x.get()) {
          instance = x;
          return true;
        }
      }
      return false;
    };
    _iterateFactories(service_name, f);
    if (instance.get())
      return instance;
    // TODO: améliorer le message
    ARCANE_FATAL("Can not create instance");
    return instance;
  }

  String printFactories() const;

  void fillWithGlobalFactories();

 private:
  Impl* m_p = nullptr;

 private:
  void _add(IInjectedInstance* instance);

  // Itère sur la lambda et s'arrête dès que cette dernière retourne \a true
  template <typename Lambda> void
  _iterateInstances(const std::type_info& t_info, const String& instance_name, const Lambda& lambda)
  {
    bool has_no_name = instance_name.empty();
    Integer n = _nbValue();
    for (Integer i = 0; i < n; ++i) {
      IInjectedInstance* ii = _value(i);
      if (!ii->hasTypeInfo(t_info))
        continue;
      if (has_no_name || ii->hasName(instance_name)) {
        if (lambda(ii))
          return;
      }
    }
  }
  Integer _nbValue() const;
  IInjectedInstance* _value(Integer i) const;

  /*!
   * \brief Itère sur la lambda et s'arrête dès que cette dernière retourne \a true.
   * Si \a factory_name n'est pas nul, seules les fabriques pour lequelles
   * FactoryInfo::hasName(factory_name) est vrai sont utilisées.
   */
  template <typename Lambda> void
  _iterateFactories(const String& factory_name, const Lambda& lambda) const
  {
    bool has_no_name = factory_name.empty();
    Integer n = _nbFactory();
    for (Integer i = 0; i < n; ++i) {
      impl::IInstanceFactory* f = _factory(i);
      if (has_no_name || f->factoryInfo()->hasName(factory_name)) {
        if (lambda(f))
          return;
      }
    }
  }
  Integer _nbFactory() const;
  impl::IInstanceFactory* _factory(Integer i) const;

  // Spécialisation pour les références
  template <typename InterfaceType> Ref<InterfaceType>
  _getRef(const String& instance_name)
  {
    using InjectedType = impl::IInjectedRefInstanceT<InterfaceType>;
    InjectedType* t = nullptr;
    auto f = [&](IInjectedInstance* v) -> bool {
      t = dynamic_cast<InjectedType*>(v);
      return t;
    };
    _iterateInstances(typeid(Ref<InterfaceType>), instance_name, f);
    if (t)
      return t->instance();
    // TODO: faire un fatal ou créer l'instance
    ARCANE_THROW(NotImplementedException, "Create Ref<InterfaceType> from factory");
  }

  template <typename Type> Type
  _getValue(const String& instance_name)
  {
    using InjectedType = impl::IInjectedValueInstance<Type>;
    InjectedType* t = nullptr;
    auto f = [&](IInjectedInstance* v) -> bool {
      t = dynamic_cast<InjectedType*>(v);
      return t;
    };
    _iterateInstances(typeid(Type), instance_name, f);
    if (t)
      return t->instance();
    ARCANE_FATAL("Can not find value for type");
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::DependencyInjection

namespace Arcane::DependencyInjection::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe permettant d'enregistrer un constructeur pour créer un objet
 * via un Injector.
 * \a ConstructorArgsType est un `std::tuple` des arguments du constructeur
 */
template <typename... Args>
class ConstructorRegisterer
{
 public:
  using ArgsType = std::tuple<Args...>;

  static constexpr bool isConstructor()
  {
    return true;
  }

  ConstructorRegisterer() {}

  template <std::size_t I>
  static auto _get(Injector& i) -> std::tuple_element_t<I, ArgsType>
  {
    using SelectedType = std::tuple_element_t<I, ArgsType>;
    std::cout << "RETURN _GET(I=" << I << ")\n";
    return i.get<SelectedType>();
  }

  ArgsType createTuple(Injector& i)
  {
    // TODO: supporter plus d'arguments ou passer à des 'variadic templates'
    constexpr int tuple_size = std::tuple_size<ArgsType>();
    static_assert(tuple_size < 3, "Too many arguments for createTuple (max=2)");
    if constexpr (tuple_size == 0) {
      return ArgsType();
    }
    else if constexpr (tuple_size == 1) {
      return ArgsType(_get<0>(i));
    }
    else if constexpr (tuple_size == 2) {
      return ArgsType(_get<0>(i), _get<1>(i));
    }
    // Ne devrait pas arriver mais on ne sais jamais.
    ARCANE_FATAL("Too many arguments for createTuple n={0} max=2", tuple_size);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Fabrique pour le type \a ConcreteType pour l'interface \a InterfaceType
 * via le constructeur \a ConstructorType.
 */
template <typename InterfaceType, typename ConcreteType, typename ConstructorType>
class Concrete3Factory
: public IConcreteFactory<InterfaceType>
{
  using Args = typename ConstructorType::ArgsType;

 public:
  Ref<InterfaceType> createReference(Injector& injector) override
  {
    ConstructorType ct;
    ConcreteType* st = _create(ct.createTuple(injector));
    return makeRef<InterfaceType>(st);
  }
  ConcreteFactoryTypeInfo concreteFactoryInfo() const override
  {
    return ConcreteFactoryTypeInfo::create<InterfaceType, ConcreteType, ConstructorType>();
  }
  Int32 nbConstructorArg() const override
  {
    return std::tuple_size<Args>();
  }

 private:
  /*!
   * Créé une instance du service à partir des arguments sous forme d'un std::tuple.
   *
   * \todo Regarder si on ne peut pas utiliser std::make_from_tuple()
   */
  ConcreteType* _create(const Args&& tuple_args)
  {
    ConcreteType* st = std::apply([](auto&&... args) -> ConcreteType* { return new ConcreteType(args...); }, tuple_args);
    return st;
    }
  };

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe permettant d'enregistrer la liste des interfaces d'un service.
 * \a Interfaces contient la liste des interfaces implémentées par le service
 */
template <typename... Interfaces>
class InterfaceListRegisterer
{
 public:
  /*!
   * \brief Enregistre une fabrique.
   *
   * Enregistre pour chaque interface de \a Interfaces une fabrique pour
   * créer une instance de \a ConcreteType via le constructeur \a ConstructorType
   */
  template <typename ConcreteType, typename ConstructorType> void
  registerFactory(FactoryInfo* si)
  {
    _registerFactory<ConcreteType, ConstructorType, Interfaces...>(si);
  }

 private:
  template <typename ConcreteType, typename ConstructorType,
            typename InterfaceType, typename... OtherInterfaces>
  void
  _registerFactory(FactoryInfo* fi)
  {
    auto* factory = new Concrete3Factory<InterfaceType, ConcreteType, ConstructorType>();
    fi->addFactory(makeRef<IInstanceFactory>(new InstanceFactory<InterfaceType>(fi, factory)));
    // Applique récursivement pour les autres interfaces si nécessaire
    if constexpr (sizeof...(OtherInterfaces) > 0)
      _registerFactory<ConcreteType, ConstructorType, OtherInterfaces...>(fi);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 */
template <typename ServiceType, typename ConstructorType>
class Concrete2Factory
{
  using Args = typename ConstructorType::ArgsType;

 public:
  ServiceType* createFromInjector(Injector& i)
  {
    ConstructorType ct;
    auto x = ct.createTuple(i);
    return create(x);
  }

  ServiceType* create(const Args& tuple_args)
  {
    ServiceType* st = std::apply([](auto&&... args) -> ServiceType* { return new ServiceType(args...); }, tuple_args);
    return st;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe permettant de créer et d'enregistrer les fabriques pour un service.
 */
template <typename ServiceType>
class ServiceAllInterfaceRegisterer
{
 private:
  //! Surcharge pour 1 interface
  template <typename InterfaceType> static void
  _create(FactoryInfo* si, const InterfaceType& i1)
  {
    if constexpr (InterfaceType::isConstructor()) {
      Concrete2Factory<ServiceType, InterfaceType> c2f;
      ARCANE_UNUSED(si);
      ARCANE_UNUSED(c2f);
    }
    else
      i1.template registerToFactoryInfo<ServiceType>(si);
  }

  //! Surcharge pour 2 interfaces ou plus
  template <typename I1, typename I2, typename... OtherInterfaces>
  static void _create(FactoryInfo* si, const I1& i1, const I2& i2, const OtherInterfaces&... args)
  {
    _create<I1>(si, i1);
    // Applique la récursivité sur les types restants
    _create<I2, OtherInterfaces...>(si, i2, args...);
  }

 public:
  //! Enregistre dans le service les fabriques pour les interfacs \a Interfaces
  template <typename... Interfaces> static void
  registerProviderInfo(FactoryInfo* si, const Interfaces&... args)
  {
    //si->setSingletonFactory(new Internal::SingletonServiceFactory<ServiceType,typename Interfaces::Interface ... >(si));
    _create(si, args...);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \a ServiceType type de l'instance à créér
 * \a InterfaceList listes des interfaces qu'implémente \a ServiceType
 */
template <typename ConcreteType, typename InterfaceList>
class InjectionRegisterer
{
 public:
  //! Enregistre dans le service les fabriques correspondentesd aux constructeurs \a Constructors
  template <typename... Constructors> void
  registerProviderInfo(FactoryInfo* si, const Constructors&... args)
  {
    _create(si, args...);
  }

 private:
  // TODO: Créér l'instance de 'FactoryInfo' dans le constructeur
  InterfaceList m_interface_list;

 private:
  //! Surcharge pour 1 constructeur
  template <typename ConstructorType> void
  _create(FactoryInfo* si, const ConstructorType&)
  {
    m_interface_list.template registerFactory<ConcreteType, ConstructorType>(si);
  }

  //! Surcharge pour 2 constructeurs ou plus
  template <typename C1, typename C2, typename... OtherConstructors>
  void _create(FactoryInfo* si, const C1& c1, const C2& c2, const OtherConstructors&... args)
  {
    _create<C1>(si, c1);
    // Applique la récursivité sur les types restants
    _create<C2, OtherConstructors...>(si, c2, args...);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::DependencyInjection::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCANE_DI_CONSTRUCTOR(...) \
  ::Arcane::DependencyInjection::impl::ConstructorRegisterer<__VA_ARGS__>()

#define ARCANE_DI_EMPTY_CONSTRUCTOR(...) \
  ::Arcane::DependencyInjection::impl::ConstructorRegisterer()

#define ARCANE_DI_SERVICE_INTERFACE(ainterface) \
  ::Arcane::DependencyInjection::impl::ServiceInterfaceRegisterer<ainterface>(#ainterface)

#define ARCANE_DI_SERVICE_INTERFACE_NS(ainterface_ns, ainterface) \
  ::Arcane::DependencyInjection::impl::ServiceInterfaceRegisterer<ainterface_ns ::ainterface>(#ainterface_ns, #ainterface)

// TODO: garantir au moins une interface

#define ARCANE_DI_INTERFACES(...) \
  ::Arcane::DependencyInjection::impl::InterfaceListRegisterer<__VA_ARGS__>

#define ARCANE_DI_REGISTER_PROVIDER(t_class, t_provider_property, ...) \
  namespace \
  { \
    Arcane::DependencyInjection::impl::FactoryInfo* \
    ARCANE_JOIN_WITH_LINE(arcaneCreateDependencyInjectionProviderInfo##t_class)(const Arcane::DependencyInjection::ProviderProperty& property) \
    { \
      auto* si = Arcane::DependencyInjection::impl::FactoryInfo::create(property, __FILE__, __LINE__); \
      Arcane::DependencyInjection::impl::ServiceAllInterfaceRegisterer<t_class>::registerProviderInfo(si, __VA_ARGS__); \
      return si; \
    } \
  } \
  Arcane::DependencyInjection::impl::GlobalRegisterer ARCANE_EXPORT ARCANE_JOIN_WITH_LINE(globalServiceRegisterer##aclass)(&ARCANE_JOIN_WITH_LINE(arcaneCreateDependencyInjectionProviderInfo##t_class), t_provider_property)

#define ARCANE_DI_REGISTER_PROVIDER2(t_class, t_provider_property, t_interfaces, ...) \
  namespace \
  { \
    Arcane::DependencyInjection::impl::FactoryInfo* \
    ARCANE_JOIN_WITH_LINE(arcaneCreateDependencyInjectionProviderInfo##t_class)(const Arcane::DependencyInjection::ProviderProperty& property) \
    { \
      auto* si = Arcane::DependencyInjection::impl::FactoryInfo::create(property, __FILE__, __LINE__); \
      Arcane::DependencyInjection::impl::InjectionRegisterer<t_class, t_interfaces> injection_registerer; \
      injection_registerer.registerProviderInfo(si, __VA_ARGS__); \
      return si; \
    } \
  } \
  Arcane::DependencyInjection::impl::GlobalRegisterer ARCANE_EXPORT ARCANE_JOIN_WITH_LINE(globalServiceRegisterer##aclass)(&ARCANE_JOIN_WITH_LINE(arcaneCreateDependencyInjectionProviderInfo##t_class), t_provider_property)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

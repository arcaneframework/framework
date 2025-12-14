// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DependencyInjection.h                                       (C) 2000-2025 */
/*                                                                           */
/* Types et fonctions pour gérer le pattern 'DependencyInjection'.           */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_INTERNAL_DEPENDENCYINJECTION_H
#define ARCCORE_BASE_INTERNAL_DEPENDENCYINJECTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * NOTE: API en cours de définition.
 * Ne pas utiliser en dehors de Arccore/Arcane
 */
#include "arccore/base/Ref.h"
#include "arccore/base/ExternalRef.h"
#include "arccore/base/GenericRegisterer.h"
#include "arccore/base/ReferenceCounterImpl.h"

//TODO Mettre le lancement des exceptions dans le '.cc'
#include "arccore/base/NotImplementedException.h"

#include <tuple>
#include <typeinfo>

// TODO: Améliorer les messages d'erreurs en cas d'échec de l'injection
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
class FactoryInfoImpl;
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

class ARCCORE_BASE_EXPORT IInjectedInstance
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
class ARCCORE_BASE_EXPORT ProviderProperty
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
/*!
 * \brief Informations sur les types d'une fabrique.
 *
 * Cela permet d'afficher par exemple en cas d'erreur les informations sur
 * l'interface concernée, le type concret qu'on souhaité créer et les
 * paramètres du constructeur.
 */
class ARCCORE_BASE_EXPORT ConcreteFactoryTypeInfo
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
class ARCCORE_BASE_EXPORT InjectedInstanceRef
{
  typedef Ref<IInjectedInstance> RefType;

 private:

  explicit InjectedInstanceRef(const RefType& r)
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
class ARCCORE_BASE_EXPORT IInstanceFactory
{
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();

 protected:

  virtual ~IInstanceFactory() = default;

 public:

  virtual InjectedInstanceRef createGenericReference(Injector& injector, const String& name) = 0;
  virtual const FactoryInfoImpl* factoryInfoImpl() const = 0;
  virtual ConcreteFactoryTypeInfo concreteFactoryInfo() const = 0;
  virtual Int32 nbConstructorArg() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Informations pour une fabrique.
 */
class ARCCORE_BASE_EXPORT FactoryInfo
{
  friend class Arcane::DependencyInjection::Injector;

 private:

  explicit FactoryInfo(const ProviderProperty& property);

 public:

  static FactoryInfo create(const ProviderProperty& property,
                            [[maybe_unused]] const char* file_name,
                            [[maybe_unused]] int line_number)
  {
    return FactoryInfo(property);
  }
  void addFactory(Ref<IInstanceFactory> f);
  bool hasName(const String& str) const;
  const FactoryInfoImpl* _impl() const { return m_p.get(); }

 private:

  std::shared_ptr<FactoryInfoImpl> m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief internal.
 * \brief Classe de base pour une fabrique.
 *
 * Cette classe s'utiliser via un ReferenceCounter pour gérer sa destruction.
 */
class ARCCORE_BASE_EXPORT AbstractInstanceFactory
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
  // NOTE: On ne conserve pas une instance de 'FactoryInfo'
  // mais uniquement son implémentation pour éviter des références croisées
  // avec le std::shared_ptr.

 public:

  InstanceFactory(const FactoryInfoImpl* si, IConcreteFactory<InterfaceType>* sub_factory)
  : m_factory_info_impl(si)
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

  const FactoryInfoImpl* factoryInfoImpl() const override
  {
    return m_factory_info_impl;
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

  const FactoryInfoImpl* m_factory_info_impl = nullptr;
  IConcreteFactory<InterfaceType>* m_sub_factory = nullptr;

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

class ARCCORE_BASE_EXPORT IConcreteFactoryBase
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

class ARCCORE_BASE_EXPORT GlobalRegisterer
: public GenericRegisterer<GlobalRegisterer>
{
  using BaseClass = GenericRegisterer<GlobalRegisterer>;
  static BaseClass::Info m_global_registerer_info;

 public:

  static GenericRegisterer<GlobalRegisterer>::Info& registererInfo()
  {
    return m_global_registerer_info;
  }

 public:

  typedef FactoryInfo (*FactoryCreateFunc)(const ProviderProperty& property);

 public:

  /*!
   * \brief Crée en enregistreur pour le service \a name et la fonction \a func.
   *
   * Ce constructeur est utilisé pour enregistrer un service.
   */
  GlobalRegisterer(FactoryCreateFunc func, const ProviderProperty& property) noexcept;

 public:

  FactoryCreateFunc infoCreatorWithPropertyFunction() { return m_factory_create_func; }

  //! Nom du service
  const char* name() { return m_name; }

  const ProviderProperty& property() const { return m_factory_property; }

 private:

  FactoryCreateFunc m_factory_create_func = nullptr;
  const char* m_name = nullptr;
  ProviderProperty m_factory_property;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::DependencyInjection::impl

namespace Arcane::DependencyInjection
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Injecteur
 */
class ARCCORE_BASE_EXPORT Injector
{
  class Impl;
  using FactoryFilterFunc = bool (*)(impl::IInstanceFactory*);

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
  /*!
   * \brief Interface d'un fonctor pour appliqué à chaque fabrique.
   */
  class IFactoryVisitorFunctor
  {
   public:

    virtual ~IFactoryVisitorFunctor() = default;
    virtual bool execute(impl::IInstanceFactory* f) = 0;
  };

  template <typename Lambda> class FactoryVisitorFunctor
  : public IFactoryVisitorFunctor
  {
   public:

    FactoryVisitorFunctor(Lambda& lambda)
    : m_lambda(lambda)
    {}

   public:

    virtual bool execute(impl::IInstanceFactory* f) { return m_lambda(f); }

   private:

    Lambda& m_lambda;
  };

  /*!
   * \brief Interface d'un fonctor pour appliqué à chaque fabrique.
   */
  class IInstanceVisitorFunctor
  {
   public:

    virtual ~IInstanceVisitorFunctor() = default;
    virtual bool execute(IInjectedInstance* v) = 0;
  };

  template <typename Lambda> class InstanceVisitorFunctor
  : public IInstanceVisitorFunctor
  {
   public:

    InstanceVisitorFunctor(Lambda& lambda)
    : m_lambda(lambda)
    {}

   public:

    virtual bool execute(IInjectedInstance* v) { return m_lambda(v); }

   private:

    Lambda& m_lambda;
  };

 public:

  Injector();
  Injector(const Injector&) = delete;
  Injector& operator=(const Injector&) = delete;
  ~Injector();

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

  /*!
   * \brief Créé une instance implémentant une interface.
   *
   * Créé et retourne une instance dont l'implémentation est
   * \a implementation_name et qui implémente l'interface \a InterfaceType.
   *
   * Si l'implémentation \a implementation_name n'est pas trouvé ou si
   * elle n'implémente pas l'interface \a InterfaceType, le comportement
   * est le suivant:
   * - si \a allow_null vaut \a true, retourne une référence nulle,
   * - si \a allow_null vaut \a false, lève une exception de type
   * FatalErrorException.
   */
  template <typename InterfaceType> Ref<InterfaceType>
  createInstance(const String& implementation_name, bool allow_null = false)
  {
    using FactoryType = impl::InstanceFactory<InterfaceType>;
    Ref<InterfaceType> instance;
    auto f = [&](impl::IInstanceFactory* v) -> bool {
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
    FactoryVisitorFunctor ff(f);
    _iterateFactories(implementation_name, &ff);
    if (instance.get() || allow_null)
      return instance;

    // Pas d'implémentation correspondante trouvée.
    // Dans ce cas on récupère la liste des implémentations valides et on les affiche dans
    // le message d'erreur.
    auto filter_func = [](impl::IInstanceFactory* v) -> bool {
      return dynamic_cast<FactoryType*>(v) != nullptr;
    };
    _printValidImplementationAndThrow(A_FUNCINFO, implementation_name, filter_func);
  }

  String printFactories() const;

  void fillWithGlobalFactories();

 private:

  Impl* m_p = nullptr;

 private:

  void _add(IInjectedInstance* instance);

  // Itère sur la lambda et s'arrête dès que cette dernière retourne \a true
  void _iterateInstances(const std::type_info& t_info, const String& instance_name,
                         IInstanceVisitorFunctor* lambda);
  size_t _nbValue() const;
  IInjectedInstance* _value(size_t i) const;

  /*!
   * \brief Itère sur les fabriques et applique le fonctor \a functor.
   *
   * On s'arrête dès qu'un appel à functor retourne \a true.
   *
   * Si \a factory_name n'est pas nul, seules les fabriques pour lequelles
   * FactoryInfo::hasName(factory_name) est vrai sont utilisées.
   */
  void _iterateFactories(const String& factory_name, IFactoryVisitorFunctor* functor) const;
  size_t _nbFactory() const;
  impl::IInstanceFactory* _factory(size_t i) const;

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
    InstanceVisitorFunctor ff(f);
    _iterateInstances(typeid(Ref<InterfaceType>), instance_name, &ff);
    if (t)
      return t->instance();
    // TODO: faire un fatal ou créer l'instance
    ARCCORE_THROW(NotImplementedException, "Create Ref<InterfaceType> from factory");
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
    InstanceVisitorFunctor ff(f);
    _iterateInstances(typeid(Type), instance_name, &ff);
    if (t)
      return t->instance();
    _doError(A_FUNCINFO, "Can not find value for type");
  }
  [[noreturn]] void _printValidImplementationAndThrow(const TraceInfo& ti,
                                                      const String& implementation_name,
                                                      FactoryFilterFunc filter_func);
  [[noreturn]] void _doError(const TraceInfo& ti, const String& message);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::DependencyInjection

namespace Arcane::DependencyInjection::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_BASE_EXPORT ConstructorRegistererBase
{
 protected:

  [[noreturn]] void _doError1(const String& message, int nb_value);
};

/*!
 * \internal
 * \brief Classe permettant d'enregistrer un constructeur pour créer un objet
 * via un Injector.
 * \a ConstructorArgsType est un `std::tuple` des arguments du constructeur
 */
template <typename... Args>
class ConstructorRegisterer
: public ConstructorRegistererBase
{
 public:

  using ArgsType = std::tuple<Args...>;

  ConstructorRegisterer() {}

  // Permet de récupérer via l'injecteur \a i le I-ème argument du tuple.
  template <std::size_t I>
  static auto _get(Injector& i) -> std::tuple_element_t<I, ArgsType>
  {
    using SelectedType = std::tuple_element_t<I, ArgsType>;
    //std::cout << "RETURN _GET(I=" << I << ")\n";
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
    _doError1("Too many arguments for createTuple n={0} max=2", tuple_size);
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
class ConcreteFactory
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
  registerFactory(FactoryInfo& si)
  {
    _registerFactory<ConcreteType, ConstructorType, Interfaces...>(si);
  }

 private:

  template <typename ConcreteType, typename ConstructorType,
            typename InterfaceType, typename... OtherInterfaces>
  void
  _registerFactory(FactoryInfo& fi)
  {
    auto* factory = new ConcreteFactory<InterfaceType, ConcreteType, ConstructorType>();
    fi.addFactory(createRef<InstanceFactory<InterfaceType>>(fi._impl(), factory));
    // Applique récursivement pour les autres interfaces si nécessaire
    if constexpr (sizeof...(OtherInterfaces) > 0)
      _registerFactory<ConcreteType, ConstructorType, OtherInterfaces...>(fi);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe permettant d'enregistrer des constructeurs pour créer un
 * type \a ConcreteType implémentant les interfaces de \a InterfaceList (qui
 * doit être du type InterfaceListRegisterer).
 */
template <typename ConcreteType, typename InterfaceList>
class InjectionRegisterer
{
 public:

  //! Enregistre dans \a si les fabriques correspondentes aux constructeurs \a Constructors
  template <typename... Constructors> void
  registerProviderInfo(FactoryInfo& si, const Constructors&... args)
  {
    _create(si, args...);
  }

 private:

  // TODO: Créér l'instance de 'FactoryInfo' dans le constructeur
  InterfaceList m_interface_list;

 private:

  //! Surcharge pour 1 constructeur
  template <typename ConstructorType> void
  _create(FactoryInfo& si, const ConstructorType&)
  {
    m_interface_list.template registerFactory<ConcreteType, ConstructorType>(si);
  }

  //! Surcharge pour 2 constructeurs ou plus
  template <typename C1, typename C2, typename... OtherConstructors>
  void _create(FactoryInfo& si, const C1& c1, const C2& c2, const OtherConstructors&... args)
  {
    _create<C1>(si, c1);
    // Applique la récursivité sur les types restants
    _create<C2, OtherConstructors...>(si, c2, args...);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::DependencyInjection::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCANE_DI_CONSTRUCTOR(...) \
  ::Arcane::DependencyInjection::impl::ConstructorRegisterer<__VA_ARGS__>()

#define ARCANE_DI_EMPTY_CONSTRUCTOR(...) \
  ::Arcane::DependencyInjection::impl::ConstructorRegisterer<>()

// TODO: garantir au moins une interface

#define ARCANE_DI_INTERFACES(...) \
  ::Arcane::DependencyInjection::impl::InterfaceListRegisterer<__VA_ARGS__>

#define ARCANE_DI_REGISTER_PROVIDER(t_class, t_provider_property, t_interfaces, ...) \
  namespace \
  { \
    Arcane::DependencyInjection::impl::FactoryInfo \
    ARCCORE_JOIN_WITH_LINE(arcaneCreateDependencyInjectionProviderInfo##t_class)(const Arcane::DependencyInjection::ProviderProperty& property) \
    { \
      auto si = Arcane::DependencyInjection::impl::FactoryInfo::create(property, __FILE__, __LINE__); \
      Arcane::DependencyInjection::impl::InjectionRegisterer<t_class, t_interfaces> injection_registerer; \
      injection_registerer.registerProviderInfo(si, __VA_ARGS__); \
      return si; \
    } \
  } \
  Arcane::DependencyInjection::impl::GlobalRegisterer ARCCORE_EXPORT ARCCORE_JOIN_WITH_LINE(globalServiceRegisterer##aclass)(&ARCCORE_JOIN_WITH_LINE(arcaneCreateDependencyInjectionProviderInfo##t_class), t_provider_property)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

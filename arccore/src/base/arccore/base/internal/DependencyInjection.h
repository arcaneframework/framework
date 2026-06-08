// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DependencyInjection.h                                       (C) 2000-2025 */
/*                                                                           */
/* Types and functions to manage the 'DependencyInjection' pattern.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_INTERNAL_DEPENDENCYINJECTION_H
#define ARCCORE_BASE_INTERNAL_DEPENDENCYINJECTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * NOTE: API is currently being defined.
 * Do not use outside of Arccore/Arcane
 */
#include "arccore/base/Ref.h"
#include "arccore/base/ExternalRef.h"
#include "arccore/base/GenericRegisterer.h"
#include "arccore/base/ReferenceCounterImpl.h"

//TODO Put the exception throwing in the '.cc'
#include "arccore/base/NotImplementedException.h"

#include <tuple>
#include <typeinfo>

// TODO: Improve error messages in case of injection failure
// TODO: Add methods to display the type in case of error (use A_FUNC_INFO)
// TODO: Add verbose mode
// TODO: Support multiple constructors and do not fail if the first
// does not work
// TODO: Support external instances (such as those created in C#).

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
 * \warning This class is used in global constructors/destructors
 * and must not perform allocation/deallocation or use types that do.
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
 * \brief Information about the types of a factory.
 *
 * This allows displaying, for example, in case of an error, information about
 * the interface concerned, the concrete type that we wanted to create, and the
 * constructor parameters.
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
 * \brief Typed interface managing the instance of a service.
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
 * \brief Typed interface managing the instance of a service.
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
 * \brief Typed interface managing an instance
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
 * \brief Typed interface managing the instance of a service.
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
 * \brief Reference to an injected instance.
 *
 * This class is managed via a reference counter in the manner
 * of the std::shared_ptr class.
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
 * \brief Factory for an instance encapsulated by a reference (i.e Ref<T>).
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
 * \brief Information for a factory.
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
 * \brief Base class for a factory.
 *
 * This class uses a ReferenceCounter to manage its destruction.
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
  // NOTE: We do not keep an instance of 'FactoryInfo'
  // but only its implementation to avoid cross-references
  // with std::shared_ptr.

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
 * \brief Interface of a functor for creating a service instance
 * corresponding to the interface \a InterfaceType.
 */
template <typename InterfaceType>
class IConcreteFactory
: public IConcreteFactoryBase
{
 public:

  virtual ~IConcreteFactory() = default;

 public:

  //! Creates an instance of the service.
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
   * \brief Registers the service \a name using the function \a func.
   *
   * This constructor is used to register a service.
   */
  GlobalRegisterer(FactoryCreateFunc func, const ProviderProperty& property) noexcept;

 public:

  FactoryCreateFunc infoCreatorWithPropertyFunction() { return m_factory_create_func; }

  //! Service name
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
 * \brief Injector
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

  //! Specialization for 'Ref'.
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
   * \brief Interface of a functor to apply to each factory.
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
   * \brief Interface of a functor to apply to each factory.
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
   * \brief Creates an instance implementing an interface.
   *
   * Creates and returns an instance whose implementation is
   * \a implementation_name and which implements the interface \a InterfaceType.
   *
   * If the implementation \a implementation_name is not found or if
   * it does not implement the interface \a InterfaceType, the behavior
   * is as follows:
   * - if \a allow_null equals \a true, returns a null reference,
   * - if \a allow_null equals \a false, throws an exception of type
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

    // No corresponding implementation found.
    // In this case, we retrieve the list of valid implementations
    // and display them in the error message.
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

  // Iterates over the lambda and stops as soon as it returns \a true
  void _iterateInstances(const std::type_info& t_info, const String& instance_name,
                         IInstanceVisitorFunctor* lambda);
  size_t _nbValue() const;
  IInjectedInstance* _value(size_t i) const;

  /*!
   * \brief Iterates over the factories and applies the functor \a functor.
   *
   * It stops as soon as a call to functor returns \a true.
   *
   * If \a factory_name is not null, only factories for which
   * FactoryInfo::hasName(factory_name) is true are used.
   */
  void _iterateFactories(const String& factory_name, IFactoryVisitorFunctor* functor) const;
  size_t _nbFactory() const;
  impl::IInstanceFactory* _factory(size_t i) const;

  // Specialization for references
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
    // TODO: throw fatal or create the instance
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
 * \brief Class allowing registration of a constructor to create an object
 * via an Injector.
 * \a ConstructorArgsType is a `std::tuple` of the constructor arguments
 */
template <typename... Args>
class ConstructorRegisterer
: public ConstructorRegistererBase
{
 public:

  using ArgsType = std::tuple<Args...>;

  ConstructorRegisterer() {}

  // Allows retrieving the I-th argument of the tuple via the injector \a i.
  template <std::size_t I>
  static auto _get(Injector& i) -> std::tuple_element_t<I, ArgsType>
  {
    using SelectedType = std::tuple_element_t<I, ArgsType>;
    //std::cout << "RETURN _GET(I=" << I << ")\n";
    return i.get<SelectedType>();
  }

  ArgsType createTuple(Injector& i)
  {
    // TODO: support more arguments or switch to 'variadic templates'
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
    // Should not happen but you never know.
    _doError1("Too many arguments for createTuple n={0} max=2", tuple_size);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Factory for the type \a ConcreteType for the interface \a InterfaceType
 * via the constructor \a ConstructorType.
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
   * Creates an instance of the service from arguments in the form of a std::tuple.
   *
   * \todo See if we can use std::make_from_tuple()
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
 * \brief Class allowing registration of the list of interfaces of a service.
 * \a Interfaces contains the list of interfaces implemented by the service
 */
template <typename... Interfaces>
class InterfaceListRegisterer
{
 public:

  /*!
   * \brief Registers a factory.
   *
   * Registers a factory for each interface in \a Interfaces to
   * create an instance of \a ConcreteType via the constructor \a ConstructorType
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
    // Recursively applies for other interfaces if necessary
    if constexpr (sizeof...(OtherInterfaces) > 0)
      _registerFactory<ConcreteType, ConstructorType, OtherInterfaces...>(fi);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Class allowing registration of constructors to create a
 * type \a ConcreteType implementing the interfaces of \a InterfaceList (which
 * must be of type InterfaceListRegisterer).
 */
template <typename ConcreteType, typename InterfaceList>
class InjectionRegisterer
{
 public:

  //! Registers in \a si the factories corresponding to the constructors \a Constructors
  template <typename... Constructors> void
  registerProviderInfo(FactoryInfo& si, const Constructors&... args)
  {
    _create(si, args...);
  }

 private:

  // TODO: Create the 'FactoryInfo' instance in the constructor
  InterfaceList m_interface_list;

 private:

  //! Overload for 1 constructor
  template <typename ConstructorType> void
  _create(FactoryInfo& si, const ConstructorType&)
  {
    m_interface_list.template registerFactory<ConcreteType, ConstructorType>(si);
  }

  //! Overload for 2 or more constructors
  template <typename C1, typename C2, typename... OtherConstructors>
  void _create(FactoryInfo& si, const C1& c1, const C2& c2, const OtherConstructors&... args)
  {
    _create<C1>(si, c1);
    // Applies recursion on the remaining types
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

// TODO: guarantee at least one interface

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

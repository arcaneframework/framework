// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DependencyInjection.cc                                      (C) 2000-2025 */
/*                                                                           */
/* Types et fonctions pour gérer le pattern 'DependencyInjection'.           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/internal/DependencyInjection.h"

#include "arcane/utils/UniqueArray.h"
#include "arcane/utils/ExternalRef.h"
#include "arcane/utils/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::DependencyInjection
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Injector::Impl
{
 public:

  class InstanceInfo
  {
   public:

    InstanceInfo(IInjectedInstance* instance, Int32 index)
    : m_instance(instance)
    , m_index(index)
    {}

   public:

    IInjectedInstance* m_instance = nullptr;
    Int32 m_index = 0;
  };

 public:

  ~Impl()
  {
    for (Integer i = 0, n = m_instance_list.size(); i < n; ++i)
      delete m_instance_list[i].m_instance;
    m_instance_list.clear();
  }

 public:

  void addInstance(IInjectedInstance* instance)
  {
    Int32 index = m_instance_list.size();
    m_instance_list.add(InstanceInfo{ instance, index });
  }
  IInjectedInstance* instance(Int32 index) const { return m_instance_list[index].m_instance; }
  Int32 nbInstance() const { return m_instance_list.size(); }

 private:

  UniqueArray<InstanceInfo> m_instance_list;

 public:

  // Il faut conserver une instance de FactoryInfo pour éviter sa
  // destruction prématurée car les instances dans m_factories en ont besoin.
  UniqueArray<Ref<impl::IInstanceFactory>> m_factories;
  UniqueArray<impl::FactoryInfo> m_factories_info;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Injector::
Injector()
: m_p(new Impl())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Injector::
~Injector()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Injector::
_add(IInjectedInstance* instance)
{
  m_p->addInstance(instance);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer Injector::
_nbValue() const
{
  return m_p->nbInstance();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IInjectedInstance* Injector::
_value(Integer i) const
{
  return m_p->instance(i);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer Injector::
_nbFactory() const
{
  return m_p->m_factories.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

impl::IInstanceFactory* Injector::
_factory(Integer i) const
{
  return m_p->m_factories[i].get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::DependencyInjection::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstructorRegistererBase::
_doError1(const String& message, int nb_value)
{
  ARCANE_FATAL(message,nb_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class FactoryInfoImpl
{
 public:

  FactoryInfoImpl(const ProviderProperty& property)
  : m_property(property)
  , m_name(property.name())
  {
  }

 public:

  bool hasName(const String& str) const { return str == m_name; }
  void fillWithImplementationNames(Array<String>& names) const { names.add(m_name); }

 public:

  const ProviderProperty m_property;
  UniqueArray<Ref<IInstanceFactory>> m_factories;
  String m_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FactoryInfo::
FactoryInfo(const ProviderProperty& property)
: m_p(std::make_shared<FactoryInfoImpl>(property))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FactoryInfo::
addFactory(Ref<IInstanceFactory> f)
{
  m_p->m_factories.add(f);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool FactoryInfo::
hasName(const String& str) const
{
  return m_p->hasName(str);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GlobalRegisterer::
GlobalRegisterer(FactoryCreateFunc func, const ProviderProperty& property) noexcept
: m_factory_create_func(func)
, m_factory_property(property)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GenericRegisterer<GlobalRegisterer>::Info GlobalRegisterer::m_global_registerer_info;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace impl

namespace Arcane::DependencyInjection
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Injector::
fillWithGlobalFactories()
{
  impl::GlobalRegisterer* g = impl::GlobalRegisterer::firstRegisterer();
  Integer i = 0;
  while (g) {
    auto func = g->infoCreatorWithPropertyFunction();
    if (func) {
      impl::FactoryInfo fi = (*func)(g->property());
      m_p->m_factories_info.add(fi);
      m_p->m_factories.addRange(fi.m_p->m_factories);
    }

    g = g->nextRegisterer();
    ++i;
    if (i > 100000)
      ARCANE_FATAL("Infinite loop in DependencyInjection global factories");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String Injector::
printFactories() const
{
  std::ostringstream ostr;
  Integer index = 0;
  auto f = [&](impl::IInstanceFactory* v) -> bool {
    const impl::ConcreteFactoryTypeInfo& cfi = v->concreteFactoryInfo();
    ostr << "I=" << index << " " << typeid(v).name()
         << "\n  interface=" << cfi.interfaceTypeInfo().traceInfo()
         << "\n  concrete=" << cfi.concreteTypeInfo().traceInfo()
         << "\n  constructor=" << cfi.constructorTypeInfo().traceInfo()
         << "\n";
    ++index;
    return false;
  };
  FactoryVisitorFunctor ff(f);
  _iterateFactories(String(), &ff);
  String s = ostr.str();
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Injector::
_iterateFactories(const String& factory_name, IFactoryVisitorFunctor* functor) const
{
  // TODO: utiliser le std::type_info de l'instance qu'on souhaite pour limiter
  // les itérations

  // Il faut trouver un constructeur qui ait le même nombre d'arguments
  // que le nombre d'instances enregistrées
  bool has_no_name = factory_name.empty();
  Integer n = _nbFactory();
  Integer nb_instance = _nbValue();
  for (Integer i = 0; i < n; ++i) {
    impl::IInstanceFactory* f = _factory(i);
    Int32 nb_constructor_arg = f->nbConstructorArg();
    if (nb_constructor_arg >= 0 && nb_constructor_arg != nb_instance)
      continue;
    if (has_no_name || f->factoryInfoImpl()->hasName(factory_name)) {
      if (functor->execute(f))
        return;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Itère sur la lambda et s'arrête dès que cette dernière retourne \a true
void Injector::
_iterateInstances(const std::type_info& t_info, const String& instance_name,
                  IInstanceVisitorFunctor* lambda)
{
  bool has_no_name = instance_name.empty();
  Integer n = _nbValue();
  for (Integer i = 0; i < n; ++i) {
    IInjectedInstance* ii = _value(i);
    if (!ii->hasTypeInfo(t_info))
      continue;
    if (has_no_name || ii->hasName(instance_name)) {
      if (lambda->execute(ii))
        return;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Injector::
_doError(const TraceInfo& ti, const String& message)
{
  ARCANE_FATAL("Function: {0} : {1}", ti, message);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Injector::
_printValidImplementationAndThrow(const TraceInfo& ti,
                                  const String& implementation_name,
                                  FactoryFilterFunc filter_func)
{
  // Pas d'implémentation correspondante trouvée.
  // Dans ce cas on récupère la liste des implémentations valides et on les affiche dans
  // le message d'erreur.
  UniqueArray<String> valid_names;
  for (Int32 i = 0, n = _nbFactory(); i < n; ++i) {
    impl::IInstanceFactory* f = _factory(i);
    if (filter_func(f)) {
      f->factoryInfoImpl()->fillWithImplementationNames(valid_names);
    }
  };
  String message = String::format("No implementation named '{0}' found", implementation_name);

  // TODO: améliorer le message
  String message2;
  if (valid_names.size() == 0)
    message2 = " and no implementation is available.";
  else if (valid_names.size() == 1)
    message2 = String::format(". Valid value is: '{0}'.", valid_names[0]);
  else
    message2 = String::format(". Valid values are: '{0}'.", String::join(", ", valid_names));
  _doError(ti, message + message2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::DependencyInjection

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
ARCCORE_DEFINE_REFERENCE_COUNTED_CLASS(Arcane::DependencyInjection::impl::IInstanceFactory);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

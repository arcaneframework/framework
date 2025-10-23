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

#include "arccore/base/internal/DependencyInjection.h"

#include "arccore/base/ExternalRef.h"
#include "arccore/base/FatalErrorException.h"

#include <vector>

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

    InstanceInfo(IInjectedInstance* instance, size_t index)
    : m_instance(instance)
    , m_index(index)
    {}

   public:

    IInjectedInstance* m_instance = nullptr;
    size_t m_index = 0;
  };

 public:

  ~Impl()
  {
    for (auto& x : m_instance_list)
      delete x.m_instance;
    m_instance_list.clear();
  }

 public:

  void addInstance(IInjectedInstance* instance)
  {
    size_t index = m_instance_list.size();
    m_instance_list.push_back(InstanceInfo{ instance, index });
  }
  IInjectedInstance* instance(size_t index) const { return m_instance_list[index].m_instance; }
  size_t nbInstance() const { return m_instance_list.size(); }

 private:

  std::vector<InstanceInfo> m_instance_list;

 public:

  // Il faut conserver une instance de FactoryInfo pour éviter sa
  // destruction prématurée car les instances dans m_factories en ont besoin.
  std::vector<Ref<impl::IInstanceFactory>> m_factories;
  std::vector<impl::FactoryInfo> m_factories_info;
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

size_t Injector::
_nbValue() const
{
  return m_p->nbInstance();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IInjectedInstance* Injector::
_value(size_t i) const
{
  return m_p->instance(i);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

size_t Injector::
_nbFactory() const
{
  return m_p->m_factories.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

impl::IInstanceFactory* Injector::
_factory(size_t i) const
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
  ARCCORE_FATAL(message,nb_value);
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
  void fillWithImplementationNames(std::vector<String>& names) const { names.push_back(m_name); }

 public:

  const ProviderProperty m_property;
  std::vector<Ref<IInstanceFactory>> m_factories;
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
  m_p->m_factories.push_back(f);
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
      m_p->m_factories_info.push_back(fi);
      for (auto& x : fi.m_p->m_factories)
        m_p->m_factories.push_back(x);
      //m_p->m_factories.addRange(fi.m_p->m_factories);
    }

    g = g->nextRegisterer();
    ++i;
    if (i > 100000)
      ARCCORE_FATAL("Infinite loop in DependencyInjection global factories");
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
  size_t n = _nbFactory();
  size_t nb_instance = _nbValue();
  for (size_t i = 0; i < n; ++i) {
    impl::IInstanceFactory* f = _factory(i);
    size_t nb_constructor_arg = f->nbConstructorArg();
    if (nb_constructor_arg != nb_instance)
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
  size_t n = _nbValue();
  for (size_t i = 0; i < n; ++i) {
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
  ARCCORE_FATAL("Function: {0} : {1}", ti, message);
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
  std::vector<String> valid_names;
  for (size_t i = 0, n = _nbFactory(); i < n; ++i) {
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
  else {
    ConstArrayView<String> valid_names_view(arccoreCheckArraySize(valid_names.size()), valid_names.data());
    message2 = String::format(". Valid values are: '{0}'.", String::join(", ", valid_names_view));
  }
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

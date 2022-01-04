// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlephFactory.cc                                             (C) 2010-2019 */
/*                                                                           */
/* Fabriques pour Aleph.                                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/aleph/IAlephFactory.h"
#include "arcane/ServiceBuilder.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AlephFactory::FactoryImpl
{
 public:
  FactoryImpl(const String& name)
  : m_name(name)
  , m_initialized(false)
  {}
  ~FactoryImpl()
  {
  }

 public:
  void setFactory(Ref<IAlephFactoryImpl> factory)
  {
    m_factory = factory;
  }
  IAlephFactoryImpl* factory() { return m_factory.get(); }
  const String& name() const { return m_name; }

 private:
  Ref<IAlephFactoryImpl> m_factory;
  String m_name;

 public:
  bool m_initialized;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/******************************************************************************
 * IAlephFactory::IAlephFactory
 *****************************************************************************/
AlephFactory::
AlephFactory(IApplication* app, ITraceMng* tm)
: IAlephFactory(tm)
{
  // Liste des implémentations possibles.
  // 0 est le choix automatique qui doit aller vers une des bibliothèques suivantes:
  m_impl_map.insert(std::make_pair(1, new FactoryImpl("Sloop")));
  m_impl_map.insert(std::make_pair(2, new FactoryImpl("Hypre")));
  m_impl_map.insert(std::make_pair(3, new FactoryImpl("Trilinos")));
  m_impl_map.insert(std::make_pair(4, new FactoryImpl("Cuda")));
  m_impl_map.insert(std::make_pair(5, new FactoryImpl("PETSc")));
  ServiceBuilder<IAlephFactoryImpl> sb(app);
  // Pour chaque implémentation possible,
  // créé la fabrique correspondante si elle est disponible.
  for (auto i : m_impl_map) {
    FactoryImpl* implementation = i.second;
    const String& name = implementation->name();
    debug() << "\33[1;34m\t[AlephFactory] Adding " << name << " library..."
            << "\33[0m";
    auto factory = sb.createReference(name + "AlephFactory", SB_AllowNull);
    implementation->setFactory(factory);
  }
  debug() << "\33[1;34m\t[AlephFactory] done"
          << "\33[0m";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlephFactory::
~AlephFactory()
{
  debug() << "\33[1;34m\t[~AlephFactory] Destruction des fabriques"
          << "\33[0m";
  for (auto i : m_impl_map)
    delete i.second;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IAlephFactoryImpl* AlephFactory::
_getFactory(Integer solver_index)
{
  FactoryImplMap::const_iterator ci = m_impl_map.find(solver_index);
  if (ci == m_impl_map.end())
    ARCANE_FATAL("Invalid solver index '{0}' for aleph factory", solver_index);
  FactoryImpl* implementation = ci->second;
  IAlephFactoryImpl* factory = implementation->factory();
  if (!factory)
    throw NotSupportedException(A_FUNCINFO,
                                String::format("Implementation for '{0}' not available",
                                               implementation->name()));
  // Si la fabrique de l'implémentation considérée n'a pas
  // été initialisée, on le fait maintenant
  if (!implementation->m_initialized) {
    debug() << "\33[1;34m\t\t[_getFactory] initializing solver_index="
            << solver_index << " ..."
            << "\33[0m";
    implementation->m_initialized = true;
    factory->initialize();
  }
  return factory;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool AlephFactory::
hasSolverImplementation(Integer solver_index)
{
  FactoryImplMap::const_iterator ci = m_impl_map.find(solver_index);
  if (ci == m_impl_map.end())
    return false;
  FactoryImpl* implementation = ci->second;
  IAlephFactoryImpl* factory = implementation->factory();
  if (!factory)
    return false;
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IAlephTopology* AlephFactory::
GetTopology(AlephKernel* kernel, Integer index, Integer nb_row_size)
{
  debug() << "\33[1;34m\t\t[IAlephFactory::GetTopology] Switch=" << kernel->underlyingSolver() << "\33[0m";
  auto f = _getFactory(kernel->underlyingSolver());
  return f->createTopology(traceMng(), kernel, index, nb_row_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IAlephVector* AlephFactory::
GetVector(AlephKernel* kernel, Integer index)
{
  debug() << "\33[1;34m\t\t[AlephFactory::GetVector] Switch=" << kernel->underlyingSolver() << "\33[0m";
  auto f = _getFactory(kernel->underlyingSolver());
  return f->createVector(traceMng(), kernel, index);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IAlephMatrix* AlephFactory::
GetMatrix(AlephKernel* kernel, Integer index)
{
  debug() << "\33[1;34m\t\t[AlephFactory::GetMatrix] Switch=" << kernel->underlyingSolver() << "\33[0m";
  auto f = _getFactory(kernel->underlyingSolver());
  return f->createMatrix(traceMng(), kernel, index);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ModuleMng.cc                                                (C) 2000-2019 */
/*                                                                           */
/* Classe gérant l'ensemble des modules.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Iostream.h"
#include "arcane/utils/List.h"
#include "arcane/utils/Deleter.h"
#include "arcane/utils/String.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Ref.h"

#include "arcane/IModuleMng.h"
#include "arcane/IModule.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire des modules.
 */
class ModuleMng
: public IModuleMng
{
 public:

  ModuleMng(ISubDomain*);
  ~ModuleMng();

  void addModule(Ref<IModule>) override;
  void removeModule(Ref<IModule>) override;
  void dumpList(std::ostream&) override;
  ModuleCollection modules() const override { return m_modules; }
  void removeAllModules() override;
  bool isModuleActive(const String& name) override;
  IModule* findModule(const String& name) override;

 private:

  ModuleList m_modules; //!< Liste des modules
  std::map<String,Ref<IModule>> m_modules_map;
  IModule* _findModule(const String& name);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" IModuleMng*
arcaneCreateModuleMng(ISubDomain* sd)
{
  return new ModuleMng(sd);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ModuleMng::
ModuleMng([[maybe_unused]] ISubDomain* sd)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \todo regarder plantage MPI lorsqu'on fait le delete.
 */
ModuleMng::
~ModuleMng()
{
  removeAllModules();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleMng::
removeAllModules()
{
  m_modules.clear();
  m_modules_map.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleMng::
addModule(Ref<IModule> module)
{
  const String& module_name = module->name();
  auto iter = m_modules_map.find(module_name);
  if (iter!=m_modules_map.end())
    ARCANE_FATAL("A module named '{0}' is already registered",module_name);
  m_modules.add(module.get());
  m_modules_map.insert(std::make_pair(module_name,module));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleMng::
removeModule(Ref<IModule> module)
{
  const String& module_name = module->name();
  auto iter = m_modules_map.find(module_name);
  m_modules_map.erase(iter);

  m_modules.remove(module.get());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ModuleMng::
dumpList(std::ostream& o)
{
  o << "** ModuleMng::dump_list: " << m_modules.count();
  o << '\n';
  for( ModuleList::Enumerator i(m_modules); ++i; ){
    o << "** Module: " << (*i)->name();
    o << '\n';
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ModuleMng::
isModuleActive(const String& name)
{
  IModule* module = _findModule(name);
  if (!module)
    return false;
  return !module->disabled();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IModule* ModuleMng::
findModule(const String& name)
{
  return _findModule(name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IModule* ModuleMng::
_findModule(const String& name)
{
 auto iter = m_modules_map.find(name);
 if (iter!=m_modules_map.end())
   return iter->second.get();
 return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

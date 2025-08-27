// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ModuleFactory.h                                             (C) 2000-2025 */
/*                                                                           */
/* Manufacture des modules.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MODULEFACTORY_H
#define ARCANE_CORE_MODULEFACTORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IServiceFactory.h"
#include "arcane/core/IModuleFactory.h"
#include "arcane/core/ServiceRegisterer.h"
#include "arcane/core/ModuleBuildInfo.h"
#include "arcane/core/IServiceInfo.h"
#include "arcane/core/ModuleProperty.h"
#include "arcane/core/ServiceProperty.h"

#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Fabrique d'un module.
 */
class ARCANE_CORE_EXPORT ModuleFactory
: public IModuleFactoryInfo
{
 public:
  
  /*!
   * \brief Construit une fabrique pour un module.
   *
   * Ce constructeur est celui utilisé par les macros ARCANE_REGISTER_MODULE
   * et ARCANE_REGISTER_AXL_MODULE.
   *
   * Cette instance devient propriétaire de \a factory et la détruira
   * dans le destructeur.
   */
  ModuleFactory(Ref<IModuleFactory2> factory,bool is_autoload);
  ~ModuleFactory() override;

 public:
  
  void addReference() override;
  void removeReference() override;
  Ref<IModule> createModule(ISubDomain* parent,const MeshHandle& mesh_handle) override;
  bool isAutoload() const override { return m_is_autoload; }
  void initializeModuleFactory(ISubDomain* sub_domain) override;
  String moduleName() const override { return m_name; }
  const IServiceInfo* serviceInfo() const override;

 private:
  
  Ref<IModuleFactory2> m_factory;
  bool m_is_autoload;
  String m_name;
  std::atomic<Int32> m_nb_ref;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe implémentant IModuleFactory2.
 */
class ARCANE_CORE_EXPORT ModuleFactory2
: public IModuleFactory2
{
 public:

  ModuleFactory2(IServiceInfo* service_info,const String& name)
  : m_service_info(service_info), m_name(name)
  {
  }
  ~ModuleFactory2() override;

  String moduleName() const override
  {
    return m_name;
  }

  const IServiceInfo* serviceInfo() const override
  {
    return m_service_info;
  }

 private:

  IServiceInfo* m_service_info;
  String m_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe template de fabrique de module.
 *
 * Cette classe permet de créer un module implémenté par la classe \a ModuleType.
 */
template<class ModuleType>
class ModuleFactory2T
: public ModuleFactory2
{
 public:

  ModuleFactory2T(IServiceInfo* service_info,const String& name)
  : ModuleFactory2(service_info,name)
  {
  }
  
  Ref<IModule> createModuleInstance(ISubDomain* sd,const MeshHandle& mesh_handle) override
  {
    auto x = new ModuleType(ModuleBuildInfo(sd,mesh_handle,moduleName()));
    return makeRef<IModule>(x);
  }

  void initializeModuleFactory(ISubDomain* sd) override
  {
    ModuleType::staticInitialize(sd);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Service
 * \brief Macro pour enregistrer un module.
 *
 * L'appel est comme suit:
 *
 \code
 * ARCANE_REGISTER_MODULE(ClassName,
 *                        Arcane::ModuleProperty("ModuleName"));
 \endcode

 * Avec les paramètres suivants:
 * - \a ClassName est le nom de la classe du module,
 * - \a "ModuleName" est le nom du module.
 *
 * Par exemple, on peut avoir une utilisation comme suit:
 *
 \code
 * ARCANE_REGISTER_MODULE(ModuleSimpleHydro,
 *                        ModuleProperty("SimpleHydro"));
 \endcode
 *
 * \note Cette macro sert à enregistrer des modules qui n'ont pas de fichiers
 * \c axl associés. Si ce n'est pas le cas, il faut utiliser la macro
 * définie dans le fichier '.h' généré à partir du fichier \c axl.
 */
#define ARCANE_REGISTER_MODULE(class_name,a_module_properties) \
extern "C++" ARCANE_EXPORT Arcane::IModuleFactoryInfo*\
ARCANE_JOIN_WITH_LINE(arcaneCreateModuleFactory##class_name) (const Arcane::ModuleProperty& properties)  \
{\
  const char* module_name = properties.name();\
  Arcane::ServiceProperty sp(module_name,0);\
  auto* si = Arcane::Internal::ServiceInfo::create(sp,__FILE__,__LINE__); \
  Arcane::IModuleFactory2* mf = new Arcane::ModuleFactory2T< class_name >(si,module_name); \
  return new Arcane::ModuleFactory(Arcane::makeRef(mf), properties.isAutoload()); \
} \
Arcane::ServiceRegisterer ARCANE_EXPORT ARCANE_JOIN_WITH_LINE(globalModuleRegisterer##class_name) \
  (& ARCANE_JOIN_WITH_LINE(arcaneCreateModuleFactory##class_name),a_module_properties)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Macro pour enregistrer un module issu d'un fichier AXL.
 *
 * Cette macro est interne à Arcane et ne doit pas être utilisée directement
 */
#define ARCANE_REGISTER_AXL_MODULE(class_name,a_module_properties) \
extern "C++" ARCANE_EXPORT Arcane::IModuleFactoryInfo*\
ARCANE_JOIN_WITH_LINE(arcaneCreateModuleFactory##class_name) (const Arcane::ModuleProperty& properties)  \
{\
  const char* module_name = properties.name();\
  Arcane::ServiceProperty sp(module_name,0);\
  auto* si = Arcane::Internal::ServiceInfo::create(sp,__FILE__,__LINE__); \
  class_name :: fillServiceInfo(si);                            \
  Arcane::IModuleFactory2* mf = new Arcane::ModuleFactory2T< class_name >(si,module_name); \
  return new Arcane::ModuleFactory(Arcane::makeRef(mf),properties.isAutoload()); \
} \
Arcane::ServiceRegisterer ARCANE_EXPORT ARCANE_JOIN_WITH_LINE(globalModuleRegisterer##class_name) \
  (& ARCANE_JOIN_WITH_LINE(arcaneCreateModuleFactory##class_name),a_module_properties)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Macro pour définir un module de manière standard.
 *
 * Cette macro permet d'enregistrer automatiquement un module de
 * nom \a module_name en créant une instance de la classe \a class_name.
 *
 * \deprecated Utiliser ARCANE_REGISTER_MODULE à la place.
 */
#define ARCANE_DEFINE_STANDARD_MODULE(class_name,module_name) \
  ARCANE_REGISTER_MODULE(class_name,Arcane::ModuleProperty(#module_name))

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

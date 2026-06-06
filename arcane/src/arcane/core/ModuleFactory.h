// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ModuleFactory.h                                             (C) 2000-2025 */
/*                                                                           */
/* Module creation.                                                          */
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
 * \brief Module factory.
 */
class ARCANE_CORE_EXPORT ModuleFactory
: public IModuleFactoryInfo
{
 public:

  /*!
   * \brief Constructs a factory for a module.
   *
   * This constructor is used by the ARCANE_REGISTER_MODULE
   * and ARCANE_REGISTER_AXL_MODULE macros.
   *
   * This instance becomes the owner of \a factory and will destroy it
   * in the destructor.
   */
  ModuleFactory(Ref<IModuleFactory2> factory, bool is_autoload);
  ~ModuleFactory() override;

 public:

  void addReference() override;
  void removeReference() override;
  Ref<IModule> createModule(ISubDomain* parent, const MeshHandle& mesh_handle) override;
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
 * \brief Class implementing IModuleFactory2.
 */
class ARCANE_CORE_EXPORT ModuleFactory2
: public IModuleFactory2
{
 public:

  ModuleFactory2(IServiceInfo* service_info, const String& name)
  : m_service_info(service_info)
  , m_name(name)
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
 * \brief Template class for module factory.
 *
 * This class allows creating a module implemented by the class \a ModuleType.
 */
template <class ModuleType>
class ModuleFactory2T
: public ModuleFactory2
{
 public:

  ModuleFactory2T(IServiceInfo* service_info, const String& name)
  : ModuleFactory2(service_info, name)
  {
  }

  Ref<IModule> createModuleInstance(ISubDomain* sd, const MeshHandle& mesh_handle) override
  {
    auto x = new ModuleType(ModuleBuildInfo(sd, mesh_handle, moduleName()));
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
 * \brief Macro for registering a module.
 *
 * The call is as follows:
 *
 \code
 * ARCANE_REGISTER_MODULE(ClassName,
 *                        Arcane::ModuleProperty("ModuleName"));
 \endcode

 * With the following parameters:
 * - \a ClassName is the name of the module class,
 * - \a "ModuleName" is the name of the module.
 *
 * For example, usage can be as follows:
 *
 \code
 * ARCANE_REGISTER_MODULE(ModuleSimpleHydro,
 *                        ModuleProperty("SimpleHydro"));
 \endcode
 *
 * \note This macro is used to register modules that do not have associated files
 * \c axl files. If that is not the case, you must use the macro
 * defined in the '.h' file generated from the \c axl file.
 */
#define ARCANE_REGISTER_MODULE(class_name, a_module_properties) \
  extern "C++" ARCANE_EXPORT Arcane::IModuleFactoryInfo* \
  ARCANE_JOIN_WITH_LINE(arcaneCreateModuleFactory##class_name)(const Arcane::ModuleProperty& properties) \
  { \
    const char* module_name = properties.name(); \
    Arcane::ServiceProperty sp(module_name, 0); \
    auto* si = Arcane::Internal::ServiceInfo::create(sp, __FILE__, __LINE__); \
    Arcane::IModuleFactory2* mf = new Arcane::ModuleFactory2T<class_name>(si, module_name); \
    return new Arcane::ModuleFactory(Arcane::makeRef(mf), properties.isAutoload()); \
  } \
  Arcane::ServiceRegisterer ARCANE_EXPORT ARCANE_JOIN_WITH_LINE(globalModuleRegisterer##class_name)(&ARCANE_JOIN_WITH_LINE(arcaneCreateModuleFactory##class_name), a_module_properties)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Macro for registering a module derived from an AXL file.
 *
 * This macro is internal to Arcane and should not be used directly.
 */
#define ARCANE_REGISTER_AXL_MODULE(class_name, a_module_properties) \
  extern "C++" ARCANE_EXPORT Arcane::IModuleFactoryInfo* \
  ARCANE_JOIN_WITH_LINE(arcaneCreateModuleFactory##class_name)(const Arcane::ModuleProperty& properties) \
  { \
    const char* module_name = properties.name(); \
    Arcane::ServiceProperty sp(module_name, 0); \
    auto* si = Arcane::Internal::ServiceInfo::create(sp, __FILE__, __LINE__); \
    class_name ::fillServiceInfo(si); \
    Arcane::IModuleFactory2* mf = new Arcane::ModuleFactory2T<class_name>(si, module_name); \
    return new Arcane::ModuleFactory(Arcane::makeRef(mf), properties.isAutoload()); \
  } \
  Arcane::ServiceRegisterer ARCANE_EXPORT ARCANE_JOIN_WITH_LINE(globalModuleRegisterer##class_name)(&ARCANE_JOIN_WITH_LINE(arcaneCreateModuleFactory##class_name), a_module_properties)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Macro for defining a module in a standard way.
 *
 * This macro allows automatically registering a module of
 * name \a module_name by creating an instance of the class \a class_name.
 *
 * \deprecated Use ARCANE_REGISTER_MODULE instead.
 */
#define ARCANE_DEFINE_STANDARD_MODULE(class_name, module_name) \
  ARCANE_REGISTER_MODULE(class_name, Arcane::ModuleProperty(#module_name))

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

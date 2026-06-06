// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* EntryPoint.h                                                (C) 2000-2025 */
/*                                                                           */
/* Module entry point.                                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ENTRYPOINT_H
#define ARCANE_CORE_ENTRYPOINT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/FunctorWithAddress.h"
#include "arcane/core/IEntryPoint.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class Timer;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Information to build an entry point.
 *
 * Normally this class is not used directly. To
 * build an entry point, you must use addEntryPoint().
 */
class ARCANE_CORE_EXPORT EntryPointBuildInfo
{
 public:

  /*!
   * \brief Entry point build information.
   *
   * \param module module associated with the function
   * \param where location in the time loop where the entry point is called
   * \param property properties of the entry point (see IEntryPoint)
   * \param name name of the entry point
   * \param caller encapsulation of the method to be called.
   * \param is_destroy_caller indicates whether the entry point should destroy
   * the functor \a caller.
   *
   * Generally, \a is_destroy_caller must be \a true, otherwise the
   * memory will not be released. Note that the C# wrapping handles the functor
   * via a garbage collector, so in this case \a is_destroy_caller must
   * be \a false.
   */
  EntryPointBuildInfo(IModule* module, const String& name,
                      IFunctor* caller, const String& where, int property,
                      bool is_destroy_caller)
  : m_module(module)
  , m_name(name)
  , m_caller(caller)
  , m_where(where)
  , m_property(property)
  , m_is_destroy_caller(is_destroy_caller)
  {
  }

 public:

  IModule* module() const { return m_module; }
  const String& name() const { return m_name; }
  IFunctor* caller() const { return m_caller; }
  const String& where() const { return m_where; }
  int property() const { return m_property; }
  bool isDestroyCaller() const { return m_is_destroy_caller; }

 private:

  IModule* m_module;
  String m_name;
  IFunctor* m_caller;
  String m_where;
  int m_property;
  bool m_is_destroy_caller;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Module entry point.
 */
class ARCANE_CORE_EXPORT EntryPoint
: public IEntryPoint
{
 public:

  //! Frees resources
  ~EntryPoint() override;

 public:

  /*!
   * \brief Constructs and returns an entry point.
   *
   * The entry point is constructed with the information provided by \bi.
   * It is automatically added to the IEntryPointMng manager and should not
   * be explicitly destroyed.
   */
  static EntryPoint* create(const EntryPointBuildInfo& bi);

 public:

  String name() const override { return m_name; }
  String fullName() const override { return m_full_name; }
  ISubDomain* subDomain() const override { return m_sub_domain; }
  IModule* module() const override { return m_module; }
  void executeEntryPoint() override;
  Real totalCPUTime() const override;
  Real lastCPUTime() const override;
  Real totalElapsedTime() const override;
  Real lastElapsedTime() const override;
  Real totalTime(Timer::eTimerType type) const override;
  Real lastTime(Timer::eTimerType type) const override;
  Integer nbCall() const override { return m_nb_call; }
  String where() const override { return m_where; }
  int property() const override { return m_property; }

 private:

  ISubDomain* m_sub_domain = nullptr; //!< Sub-domain manager
  IFunctor* m_caller = nullptr; //!< Call point
  Timer* m_elapsed_timer = nullptr; //!< Entry point clock timer
  String m_name; //!< Entry point name
  String m_full_name; //!< Entry point name
  IModule* m_module = nullptr; //!< Associated module
  String m_where; //!< Call location
  int m_property = 0; //!< Entry point properties
  Integer m_nb_call = 0; //!< Number of times the entry point has been executed
  bool m_is_destroy_caller = false; //!< Indicates whether the calling functor should be destroyed.

 private:

  explicit EntryPoint(const EntryPointBuildInfo& build_info);

 public:

  EntryPoint(const EntryPoint&) = delete;
  void operator=(const EntryPoint&) = delete;

 private:

  void _getAddressForHyoda(void* = nullptr);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Template routine allowing an entry point to be referenced
 * in a module.
 *
 * The parameter \a ModuleType must be a type that derives from IModule.
 *
 * \param module Module associated with the function
 * \param func member function called by the function
 * \param where location where the entry point is called
 * \param property properties of the entry point (see IEntryPoint)
 * \param name name of the function for Arcane
 */
template <typename ModuleType> inline void
addEntryPoint(ModuleType* module, const char* name, void (ModuleType::*func)(),
              const String& where = IEntryPoint::WComputeLoop,
              int property = IEntryPoint::PNone)
{
  IFunctorWithAddress* caller = new FunctorWithAddressT<ModuleType>(module, func);
  EntryPoint::create(EntryPointBuildInfo(module, name, caller, where, property, true));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Template routine allowing an entry point to be referenced
 * in a module.
 *
 * The parameter \a ModuleType must be a type that derives from IModule.
 *
 * \param module Module associated with the function
 * \param func member function called by the function
 * \param where location where the entry point is called
 * \param property properties of the entry point (see IEntryPoint)
 * \param name name of the function for Arcane
 */
template <typename ModuleType> inline void
addEntryPoint(ModuleType* module, const String& name, void (ModuleType::*func)(),
              const String& where = IEntryPoint::WComputeLoop,
              int property = IEntryPoint::PNone)
{
  IFunctorWithAddress* caller = new FunctorWithAddressT<ModuleType>(module, func);
  EntryPoint::create(EntryPointBuildInfo(module, name, caller, where, property, true));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

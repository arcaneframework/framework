// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICaseMng.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Interface for the class managing the dataset.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ICASEMNG_H
#define ARCANE_CORE_ICASEMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ICaseMngInternal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Types of events supported by ICaseMng.
 *
 * It is possible to register for these events via the ICaseMng::observable() method.
 */
enum class eCaseMngEventType
{
  //! Event generated before reading options in phase 1
  BeginReadOptionsPhase1,
  //! Event generated before reading options in phase 2.
  BeginReadOptionsPhase2
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup CaseOption
 * \brief Case manager interface.
 *
 * This interface is managed by a reference counter and should not
 * be explicitly destroyed.
 */
class ICaseMng
{
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  // TODO: make private (start 2024)
  virtual ~ICaseMng() = default; //!< Frees resources

 public:

  //! Associated application
  virtual IApplication* application() = 0;

  //! Trace manager
  virtual ITraceMng* traceMng() = 0;

  //! Associated mesh manager
  virtual IMeshMng* meshMng() const = 0;

  //! Sub-domain manager.
  virtual ISubDomain* subDomain() = 0;

  //! XML document of the dataset (can be null if no dataset)
  virtual ICaseDocument* caseDocument() = 0;

  //! Fragment of the XML Document associated with the dataset (can be null if no dataset)
  virtual ICaseDocumentFragment* caseDocumentFragment() = 0;

  //! Associated unit system.
  virtual IPhysicalUnitSystem* physicalUnitSystem() const = 0;

  //! Reads the XML document of the dataset.
  virtual ICaseDocument* readCaseDocument(const String& filename, ByteConstArrayView bytes) = 0;

  //! Reads the dataset options corresponding to the used modules
  virtual void readOptions(bool is_phase1) = 0;

  //! Prints the option values
  virtual void printOptions() = 0;

  //! Reads the dataset tables.
  virtual void readFunctions() = 0;

 public:

  //! Registers a list of dataset options
  virtual void registerOptions(ICaseOptions*) = 0;

  //! Unregisters a list of dataset options
  virtual void unregisterOptions(ICaseOptions*) = 0;

  //! Collection of option blocks.
  virtual CaseOptionsCollection blocks() const = 0;

 public:

  //! Returns the function by name \a name or \a nullptr if none exists.
  virtual ICaseFunction* findFunction(const String& name) const = 0;

  /*!
   * \brief Returns the list of tables.
   *
   * The returned pointer is no longer valid as soon as the list of tables changes.
   */
  virtual CaseFunctionCollection functions() = 0;

  /*!
   * \brief Deletes a function.
   *
   * Deletes the function \a func. If this function is not in this list,
   * nothing is done.
   * If \a dofree is true, the delete operator is called on this function.
   */
  ARCCORE_DEPRECATED_2019("Use removeFunction(ICaseFunction*) instead.")
  virtual void removeFunction(ICaseFunction* func, bool dofree) = 0;

  /*!
   * \brief Deletes a function.
   *
   * Deletes the function \a func. If this function is not in this list,
   * nothing is done.
   */
  virtual void removeFunction(ICaseFunction* func) = 0;

  /*!
   * \brief Adds the function \a func.
   *
   * Addition can only be done during initialization. The caller remains
   * the owner of the \a func instance and must remove it via removeFunction().
   */
  ARCCORE_DEPRECATED_2019("Use addFunction(Ref<ICaseFunction>) instead.")
  virtual void addFunction(ICaseFunction* func) = 0;

  /*!
   * \brief Adds the function \a func.
   *
   * Addition can only be done during initialization.
   */
  virtual void addFunction(Ref<ICaseFunction> func) = 0;

  /*!
   * \brief Updates the options based on a time-marching table.
   *
   * For each option dependent on a marching table, updates its value
   * using the \a current_time parameter if it is a marching table with a real parameter,
   * or \a current_iteration if it is a marching table with an integer parameter.
   * If the option function has a non-zero coefficient ICaseFunction::deltatCoef(),
   * the time used is equal to current_time + coef*current_deltat.
   *
   * \param current_time time used as parameter for the function
   * \param current_deltat deltat used as parameter for the function
   * \param current_iteration iteration used as parameter for the function
   */
  virtual void updateOptions(Real current_time, Real current_deltat, Integer current_iteration) = 0;

  /*!
   * \brief Sets the way warnings are treated.
   * \sa isTreatWarningAsError().
   */
  virtual void setTreatWarningAsError(bool v) = 0;

  /*!
   * \brief Indicates whether warnings in the dataset should be treated
   * as errors and cause the code to stop.
   */
  virtual bool isTreatWarningAsError() const = 0;

  //! Sets the permission for unknown elements at the document root.
  virtual void setAllowUnkownRootElelement(bool v) = 0;

  //! Indicates whether unknown elements at the document root are allowed
  virtual bool isAllowUnkownRootElelement() const = 0;

  /*!
   * \brief Observable on the instance.
   *
   * The type of the observable is given by \a type
   */
  virtual IObservable* observable(eCaseMngEventType type) = 0;

 public:

  virtual Ref<ICaseMng> toReference() = 0;

 public:

  //! Internal implementation
  virtual ICaseMngInternal* _internalImpl() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

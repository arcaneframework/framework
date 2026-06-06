// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IVariableMngInternal.h                                      (C) 2000-2026 */
/*                                                                           */
/* Internal part of IVariableMng in Arcane.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_IVARIABLEMNG_H
#define ARCANE_CORE_INTERNAL_IVARIABLEMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of the variable manager.
 *
 * This manager contains the list of variables declared in the
 * associated subdomain \a subDomain(). It maintains the list of variables
 * and allows them to be read or written.
 */
class ARCANE_CORE_EXPORT IVariableMngInternal
{
 public:

  virtual ~IVariableMngInternal() = default; //!< Releases resources.

 public:

  /*!
   * \brief Constructs the instance members.
   *
   * The instance is not usable until this method has been
   * called. This method must be called before initialize().
   * \warning This method must only be called once.
   */
  virtual void build() = 0;

  /*!
   * \brief Initializes the instance.
   * The instance is not usable until this method has been
   * called.
   * \warning This method must only be called once.
   */
  virtual void initialize() = 0;

  //! Removes and destroys the variables managed by this manager
  virtual void removeAllVariables() = 0;

  //! Removes and destroys variables having the PInShMem property, managed
  //! by this manager.
  virtual void removeAllShMemVariables() = 0;

  //! Detaches variables associated with the mesh \a mesh.
  virtual void detachMeshVariables(IMesh* mesh) = 0;

 public:

  /*!
   * \brief Adds a reference to a variable.
   *
   * Adds the reference \a var to the manager.
   *
   * \pre var != 0
   * \pre var must not already be referenced.
   * \return the implementation associated with \a var.
   */
  virtual void addVariableRef(VariableRef* var) = 0;

  /*!
   * \brief Removes a reference to a variable.
   *
   * Removes the reference \a var from the manager.
   *
   * If \a var is not referenced by the manager, nothing is done.
   * \pre var != 0
   */
  virtual void removeVariableRef(VariableRef* var) = 0;

  /*!
   * \brief Adds a variable.
   *
   * Adds the variable \a var.
   *
   * The variable validity is not checked (void checkVariable()).
   *
   * \pre var != 0
   * \pre var must not already be referenced.
   * \return the implementation associated with \a var.
   */
  virtual void addVariable(IVariable* var) = 0;

  /*!
   * \brief Removes a variable.
   *
   * Removes the variable \a var.
   *
   * After calling this method, the variable must no longer be used.
   *
   * \pre var != 0
   * \pre var must have a single reference.
   */
  virtual void removeVariable(IVariable* var) = 0;

  /*!
   * \brief Initializes the variables.
   *
   * Iterates through the list of variables and initializes them.
   * Only variables from a used module are initialized.
   *
   * \param is_continue \a true if resuming.
   */
  virtual void initializeVariables(bool is_continue) = 0;

  /*!
   * \brief Adds the variable to the list of variables that are kept
   * until the end of execution.
   *
   * The variable will be destroyed by calling the operator delete()
   * when calling IVariableMng::removeAllVariables().
   */
  virtual void addAutoDestroyVariable(VariableRef* var) = 0;

 public:

  //! Temporary internal function to retrieve the subdomain.
  virtual ISubDomain* internalSubDomain() const = 0;

  //! Manager for accelerators
  virtual IAcceleratorMng* acceleratorMng() const = 0;

  //! Sets the accelerator manager
  virtual void setAcceleratorMng(Ref<IAcceleratorMng> v) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

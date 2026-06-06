// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableRef.h                                               (C) 2000-2025 */
/*                                                                           */
/* Class managing a reference to a variable.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_VARIABLEREF_H
#define ARCANE_CORE_VARIABLEREF_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"

#include "arcane/core/IVariable.h"
#include "arcane/core/VariableComputeFunction.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IModule;
class IVariableComputeFunction;
class VariableBuildInfo;
typedef VariableBuildInfo VariableBuilder;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Variable
 * \brief Reference to a variable.
 *
 * This class manages a reference to a variable (IVariable).
 *
 * If the variable is not associated with a module, the module() method returns 0.
 *
 * This class must be derived.
 *
 * The most derived class of this class must call _internalInit()
 * in its constructor. Only it must do so, and in the constructor
 * to ensure that the reference to the variable is valid as soon as
 * the object is constructed and that the virtual methods that must be called
 * during this initialization correspond to the instance being created.
 */
class ARCANE_CORE_EXPORT VariableRef
{
 public:

  class UpdateNotifyFunctorList;
  friend class UpdateNotifyFunctorList;

 protected:

  //! Constructs a reference to a variable with the infos \a vbi
  explicit VariableRef(const VariableBuildInfo& vbi);
  //! Copy constructor
  VariableRef(const VariableRef& from);
  //! Constructs a reference to a variable \a var
  explicit VariableRef(IVariable* var);
  //! Copy assignment operator
  VariableRef& operator=(const VariableRef& from);
  //! Default constructor
  VariableRef();

 public:

  //! Releases resources
  virtual ~VariableRef();

 public:

  //! Sub-domain associated with the variable (TODO deprecate end of 2023)
  ISubDomain* subDomain() const;

 public:

  //! Variable manager associated with the variable.
  IVariableMng* variableMng() const;

  //! Variable name
  String name() const;

 public:

  //TODO Remove virtual
  //! Variable type (Real, Integer, ...)
  virtual eDataType dataType() const;

  //! Prints the variable value
  virtual void print(std::ostream& o) const;

  //TODO Remove virtual
  //! Module associated with the variable (or nullptr, if none)
  virtual IModule* module() const { return m_module; }

  //TODO Remove virtual
  //! Variable properties
  virtual int property() const;

  //! Reference properties (internal)
  virtual int referenceProperty() const;

  //! Sets the property \a property
  virtual void setProperty(int property);

  //! Unsets the property \a property
  virtual void unsetProperty(int property);

  //! Registers the variable (internal)
  virtual void registerVariable();

  //! Unregisters the variable (internal)
  virtual void unregisterVariable();

  //! Associated variable
  IVariable* variable() const { return m_variable; }

  /*! \brief Checks if the variable is synchronized.
   * \sa IVariable::checkIfSync()
   */
  virtual Integer checkIfSync(int max_print = 0);

  /*!
   * \brief Checks if the variable has the same values on all replicas.
   * \sa IVariable::checkIfSameOnAllReplica()
   */
  virtual Integer checkIfSameOnAllReplica(int max_print = 0);

  //! Updates from the internal part
  virtual void updateFromInternal();

  //! If the variable is an array, returns its dimension, otherwise returns 0
  virtual Integer arraySize() const { return 0; }

 public:

  void setUsed(bool v) { m_variable->setUsed(v); }
  bool isUsed() const { return m_variable->isUsed(); }

  virtual void internalSetUsed(bool /*v*/) {}

 public:

  /*!
   * \brief Call stack at the time of assigning this instance.
   *
   * The stack is only accessible in verification or debug mode. If
   * not, it returns a null string.
   */
  const String& assignmentStackTrace() const { return m_assignment_stack_trace; }

 public:

  //@{ @name Tag Management
  //! Adds the tag \a tagname with the value \a tagvalue
  void addTag(const String& tagname, const String& tagvalue);
  /*! \brief Removes the tag \a tagname
   *
   * If the tag \a tagname is not in the list, nothing happens.
   */
  void removeTag(const String& tagname);
  //! \a true if the variable has the tag \a tagname
  bool hasTag(const String& tagname) const;
  //! Value of the tag \a tagname. The string is null if the tag does not exist.
  String tagValue(const String& tagname) const;
  //@}

 public:

  /*!
   * \name Dependency Management
   *
   * Operations related to variable dependency management.
   */
  //@{
  /*! \brief Recalculates the variable if necessary
   *
   * Through the dependency mechanism, this operation is called recursively
   * on all variables that the instance depends on. The recalculation function
   * computeFunction() is then called if it turns out that one of the variables
   * it depends on has been modified more recently.
   */
  void update();

  /*! \brief Indicates that the variable has just been updated.
   *
   * For correct dependency management, this property
   * must be called every time a variable update has been performed.
   */
  void setUpToDate();

  //! Time when the variable was updated
  Int64 modifiedTime();

  //! Adds \a var to the dependency list at the current time
  void addDependCurrentTime(const VariableRef& var);

  //! Adds \a var to the dependency list at the current time with trace info \a tinfo
  void addDependCurrentTime(const VariableRef& var, const TraceInfo& tinfo);

  //! Adds \a var to the dependency list at the previous time
  void addDependPreviousTime(const VariableRef& var);

  //! Adds \a var to the dependency list at the previous time with trace info \a tinfo
  void addDependPreviousTime(const VariableRef& var, const TraceInfo& tinfo);

  /*! \brief Removes \a var from the dependency list
   */
  void removeDepend(const VariableRef& var);

  /*!
   * \brief Sets the variable's recalculation function.
   *
   * If a recalculation function already existed, it is destroyed
   * and replaced by this one.
   */
  template <typename ClassType> void
  setComputeFunction(ClassType* instance, void (ClassType::*func)())
  {
    _setComputeFunction(new VariableComputeFunction(instance, func));
  }

  /*!
   * \brief Sets the variable's recalculation function.
   *
   * If a recalculation function already existed, it is destroyed
   * and replaced by this one.
   * \a tinfo contains the information allowing to know where the function is defined (for debugging)
   */
  template <typename ClassType> void
  setComputeFunction(ClassType* instance, void (ClassType::*func)(), const TraceInfo& tinfo)
  {
    _setComputeFunction(new VariableComputeFunction(instance, func, tinfo));
  }
  //@}

 public:

  //! Previous reference (or null) to variable()
  VariableRef* previousReference();

  //! Next reference (or null) to variable()
  VariableRef* nextReference();

  /*!
   * \internal
   * \brief Sets the previous reference.
   *
   * For internal use only.
   */
  void setPreviousReference(VariableRef* v);

  /*!
   * \internal
   * \brief Sets the next reference.
   *
   * For internal use only.
   */
  void setNextReference(VariableRef* v);

 public:

  static void setTraceCreation(bool v);
  static bool hasTraceCreation();

 protected:

  void _setComputeFunction(IVariableComputeFunction* v);

  /*!
   * \brief Internal initialization of the variable.
   *
   * \warning This method must <strong >obligatorily</strong > be
   * called in the derived class constructor
   * before any use of the reference.
   */
  void _internalInit(IVariable*);

  /*!
   * \brief Referenced variable.
   *
   * This method checks that a variable is properly referenced.
   */
  IVariable* _variable() const
  {
    _checkValid();
    return m_variable;
  }

 private:

  //! Associated variable
  IVariable* m_variable = nullptr;

  //! Associated module (or 0 if none)
  IModule* m_module = nullptr;

  //! \a true if the variable has been registered
  bool m_is_registered = false;

  //! Reference properties
  int m_reference_property = 0;

  //! Previous reference on \a m_variable
  VariableRef* m_previous_reference = nullptr;

  //! Next reference on \a m_variable
  VariableRef* m_next_reference = nullptr;

  /*!
   * \brief Call stack during variable assignment.
   *
   * Used only when traces are active.
   */
  String m_assignment_stack_trace;

 protected:

  void _executeUpdateFunctors();

  bool m_has_trace = false;

 private:

  void _checkValid() const
  {
#ifdef ARCANE_CHECK
    if (!m_variable)
      _throwInvalid();
#endif
  }
  void _throwInvalid() const;
  bool _checkValidPropertyChanged(int property);
  void _setAssignmentStackTrace();

 protected:

  void _internalAssignVariable(const VariableRef& var);

 private:

  static bool m_static_has_trace_creation;
  UpdateNotifyFunctorList* m_notify_functor_list = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

//TODO: to be removed when all codes include this file directly
#include "arcane/core/VariableList.h"

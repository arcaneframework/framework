// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IVariableMng.h                                              (C) 2000-2023 */
/*                                                                           */
/* Variable manager interface.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IVARIABLEMNG_H
#define ARCANE_CORE_IVARIABLEMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IVariable;
class IVariableFilter;
class VariableInfo;
class MeshVariable;
class IModule;
class IParallelMng;
class IDataReader;
class IDataWriter;
class IObservable;
class ICheckpointReader;
class CheckpointReadInfo;
class ICheckpointWriter;
class IPostProcessorWriter;
class VariableRef;
class IMesh;
class IVariableUtilities;
class VariableStatusChangedEventArgs;
class IVariableMngInternal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Variable manager interface.
 *
 * This manager contains the list of variables declared in the
 * associated sub-domain \a subDomain(). It maintains the list of variables
 * and allows them to be read or written.
 */
class IVariableMng
{
 public:

  virtual ~IVariableMng() = default; //!< Frees resources.

 public:

  //! Sub-domain manager
  ARCCORE_DEPRECATED_2020("Do not use this method. Try to get 'ISubDomain' from another way")
  virtual ISubDomain* subDomain() =0;

  //! Associated parallelism manager
  virtual IParallelMng* parallelMng() const =0;

  //! Message manager
  virtual ITraceMng* traceMng() =0;

  /*!
   * \brief Checks a variable.
   *
   * Checks that the variable named \a name characterized by \a infos
   * is valid.
   * This is true if and only if:
   * - no variable named \a infos.name() already exists.
   * - a variable named \a infos.name() exists and
   * its type and kind match \a infos.
   *
   * If the variable is not valid, an exception is thrown.
   * 
   * This operation is used when you want to create a
   * new reference to a variable and ensure that it
   * will be valid.
   *
   * \exception ExBadVariableKindType if the variable named \a infos.name() exists
   * and its type and kind do not match those of \a infos.
   *
   * \return the variable named \a infos.name() if it exists, 0 otherwise
   */
  virtual IVariable* checkVariable(const VariableInfo& infos) =0;
  /*! \brief Generates a name for a temporary variable.
   *
   * To ensure the consistency of this name, all sub-domains
   * must call this function.
   */
  virtual String generateTemporaryVariableName() =0;
  
  //! Displays the list of variables managed by a module
  virtual void dumpList(std::ostream&,IModule*) =0;

  //! Displays the list of all variables managed by the manager
  virtual void dumpList(std::ostream&) =0;


  /*!
   * \brief Estimated size for exporting variables.
   *
   This operation estimates the number of megabytes that the
   exportation of variables \a vars will generate. If \a vars is empty, the estimation
   is based on all referenced variables.
   
   The estimation only takes into account the memory used
   by the variables and not the writer used.

   The estimation is local to the sub-domain. To obtain the total size
   of an export, you must determine the size per sub-domain
   and sum them up.

   This method is collective

   \todo use 8-byte integers or more...
   */
  virtual Real exportSize(const VariableCollection& vars) =0;

  /*!
   * \brief Observable for variables being written.
   *
   * Observers registered in this observable are called
   * before writing variables (operation writeCheckpoint(),
   * writeVariables() or writePostProcessing()).
   */
  virtual IObservable* writeObservable() =0;

  /*!
   * \brief Observable for variables being read.
   *
   * Observers registered in this observable are called
   * after reading variables (operation readVariables() or readCheckpoint()).
   */
  virtual IObservable* readObservable() =0;

  /*! \brief Writes the variables.
   *
   * Iterates through all variables managed by the manager and applies the writer
   * \a writer to them. If \a filter is not null, it is applied to each variable and
   * a variable is written only if the filter is true for that variable.
   *
   * This method is collective
   */
  virtual void writeVariables(IDataWriter* writer,IVariableFilter* filter=0) =0;

  /*!
   * \brief Exports the variables.
   *
   * Exports the variables in the list \a vars. If \a vars is
   * empty, it exports all variables in the base that are used.
   */
  virtual void writeVariables(IDataWriter* writer,const VariableCollection& vars) =0;

  /*!
   * \internal
   * \brief Writes variables for a checkpoint.
   *
   * Uses the protection service \a writer to write the variables.
   *
   * This method is collective.
   *
   * This method is internal to Arcane. Generally, writing
   * a checkpoint is done via an instance of ICheckpointMng,
   * accessible via ISubDomain::checkpointMng().
   */
  virtual void writeCheckpoint(ICheckpointWriter* writer) =0;

  /*! \brief Writes variables for post-processing.
   *
   * Uses the post-processing service \a writer to write the variables.
   * The caller must have positioned the fields of \a writer before this call,
   * notably the list of variables to be post-processed. This method
   * calls IPostProcessorWriter::notifyBeginWrite() before writing
   * and IPostProcessorWriter::notifyEndWriter() at the end.
   *
   * This method is collective.
   */
  virtual void writePostProcessing(IPostProcessorWriter* writer) =0;

  /*!
   *\brief Reads all variables.
   *
   * Iterates through all variables managed by the manager and applies the reader
   * \a reader to them. If \a filter is not null, it is applied to each variable and
   * a variable is read only if the filter is true for that variable. Variables
   * that are not read are not modified by this operation.
   *
   * \deprecated Use readVariable(IDataReader*)
   *
   * This method is collective.
   */
  virtual void readVariables(IDataReader* reader,IVariableFilter* filter=0) =0;

  /*!
   * \internal
   * \brief Reads all variables from a checkpoint.
   *
   * Reads a checkpoint using the service \a reader on all
   * variables.
   *
   * This method is collective.
   *
   * This method is internal to Arcane. Generally, reading
   * a checkpoint is done via an instance of ICheckpointMng,
   * accessible via ISubDomain::checkpointMng().
   */
  virtual void readCheckpoint(ICheckpointReader* reader) =0;

  /*!
   * \internal
   * \brief Reads all variables from a checkpoint.
   *
   * Reads a checkpoint using the information contained
   * in \a infos.
   *
   * This method is collective.
   *
   * This method is internal to Arcane. Generally, reading
   * a checkpoint is done via an instance of ICheckpointMng,
   * accessible via ISubDomain::checkpointMng().
   */
  virtual void readCheckpoint(const CheckpointReadInfo& infos) =0;

  //! Gets all variables of module \a i
  virtual void variables(VariableRefCollection v,IModule* i) =0;

  //! List of variables
  virtual VariableCollection variables() =0;

  //! List of used variables
  virtual VariableCollection usedVariables() =0;
  
  //! Notifies the manager that a variable's state has changed
  virtual void notifyUsedVariableChanged() =0;
  
  //! Returns the variable named \a name or 0 if no such name exists.
  virtual IVariable* findVariable(const String& name) =0;

  //! Returns the mesh variable named \a name or 0 if no such name exists.
  virtual IVariable* findMeshVariable(IMesh* mesh,const String& name) =0;

  //! Returns the fully qualified variable named \a name or 0 if no such name exists.
  virtual IVariable* findVariableFullyQualified(const String& name) =0;

  //! Writes statistics about variables to the stream \a ostr
  virtual void dumpStats(std::ostream& ostr,bool is_verbose) =0;

  //! Writes statistics with the writer \a writer.
  virtual void dumpStatsJSON(JSONWriter& writer) =0;

  //! Interface of associated utility functions
  virtual IVariableUtilities* utilities() const =0;

  //! Interface of the variable synchronization manager.
  virtual IVariableSynchronizerMng* synchronizerMng() const =0;

 public:

  //! \name Events
  //@{
  //! Event sent when a variable is created
  virtual EventObservable<const VariableStatusChangedEventArgs&>& onVariableAdded() =0;

  //! Event sent when a variable is destroyed
  virtual EventObservable<const VariableStatusChangedEventArgs&>& onVariableRemoved() =0;
  //@}

 public:

  /*!
   * \brief Constructs the instance members.
   *
   * The instance is not usable until this method has been
   * called. This method must be called before initialize().
   * \warning This method must only be called once.
   */
  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  virtual void build() =0;

  /*!
   * \brief Initializes the instance.
   * The instance is not usable until this method has been
   * called.
   * \warning This method must only be called once.
   */
  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  virtual void initialize() =0;

  //! Removes and destroys the variables managed by this manager
  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  virtual void removeAllVariables() =0;

  //! Detaches variables associated with the mesh \a mesh.
  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  virtual void detachMeshVariables(IMesh* mesh) =0;


  /*!
   * \brief Adds a reference to a variable.
   *
   * Adds the reference \a var to the manager.
   *
   * \pre var != 0
   * \pre var must not already be referenced.
   * \return the implementation associated with \a var.
   */
  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  virtual void addVariableRef(VariableRef* var) =0;

  /*!
   * \brief Removes a reference to a variable.
   *
   * Removes the reference \a var from the manager.
   *
   * If \a var is not referenced by the manager, nothing is done.
   * \pre var != 0
   */
  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  virtual void removeVariableRef(VariableRef* var) =0;

  /*!
   * \brief Adds a variable.
   *
   * Adds the variable \a var.
   *
   * The validity of the variable is not checked (void checkVariable()).
   *
   * \pre var != 0
   * \pre var must not already be referenced.
   * \return the implementation associated with \a var.
   */
  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  virtual void addVariable(IVariable* var) =0;

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
  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  virtual void removeVariable(IVariable* var) =0;

  /*!
   * \brief Initializes the variables.
   *
   * Iterates through the list of variables and initializes them.
   * Only variables of a used module are initialized.
   *
   * \param is_continue \a true if resuming.
   */
  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  virtual void initializeVariables(bool is_continue) =0;

 public:

  /*!
   * \internal
   * Temporary internal function to retrieve the sub-domain.
   */
  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  virtual ISubDomain* _internalSubDomain() const =0;

 public:

  //! Internal Arcane API
  virtual IVariableMngInternal* _internalApi() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITraceMngPolicy.h                                           (C) 2000-2019 */
/*                                                                           */
/* Interface for trace management policy.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_ITRACEMNGPOLICY_H
#define ARCANE_UTILS_ITRACEMNGPOLICY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface for the configuration manager of a trace manager.
 *
 * The properties defined by this class are used to initialize
 * instances of ITraceMng. Modifying a property does not affect
 * already created ITraceMngs.
 */
class ARCANE_UTILS_EXPORT ITraceMngPolicy
{
 public:
  virtual ~ITraceMngPolicy(){}
 public:
  //! Constructs the instance
  virtual void build() =0;
  /*!
   * \brief Initializes \a trace.
   *
   * If \a rank is 0, then \a trace is considered the master ITraceMng. In case of listing output, the suffix will have the value \a rank.
   */
  virtual void initializeTraceMng(ITraceMng* trace,Int32 rank) =0;

  /*!
   * \brief Initializes \a trace with information from the parent \a parent_trace.
   *
   * If file outputs are enabled, \a trace will output its information
   * into a file suffixed by \a file_suffix.
   * \a parent_trace may be null.
   */
  virtual void initializeTraceMng(ITraceMng* trace,ITraceMng* parent_trace,
                                  const String& file_suffix) =0;
  /*!
   * \brief Sets the values of the TraceClassConfig of \a trace via
   * the data contained in \a bytes.
   *
   * \a bytes is a buffer containing a character string in XML format
   * as described in the documentation \ref arcanedoc_execution_traces.
   *
   * The instances of TraceClassConfig of \a trace already registered before calling this
   * method are deleted.
   */
  virtual void setClassConfigFromXmlBuffer(ITraceMng* trace,ByteConstArrayView bytes) =0;

  /*!
   * \brief Indicates if parallelism is active.
   *
   * This property is set by the application during initialization.
   */
  virtual void setIsParallel(bool v) =0;
  virtual bool isParallel() const =0;

  /*!
   * \brief Indicates if debug outputs are active.
   *
   * This property is set by the application during initialization.
   */
  virtual void setIsDebug(bool v) =0;
  virtual bool isDebug() const =0;

  /*!
   * \brief Indicates if all ranks output traces to a file in parallel.
   */
  virtual void setIsParallelOutput(bool v) =0;
  virtual bool isParallelOutput() const =0;

  /*!
   * \brief Verbosity level for the standard output stream (stdout).
   *
   * This property is used when calling initializeTraceMng()
   * to set the verbosity level of standard outputs.
   */
  virtual void setStandardOutputVerbosityLevel(Int32 level) =0;
  virtual Int32 standardOutputVerbosityLevel() const =0;

  /*!
   * \brief Verbosity level.
   *
   * This property is used when calling initializeTraceMng()
   * to set the verbosity level.
   */
  virtual void setVerbosityLevel(Int32 level) =0;
  virtual Int32 verbosityLevel() const =0;

  /*!
   * \brief Indicates if a master ITraceMng outputs traces to a file
   * in addition to standard output.
   *
   * This property defaults to \a false.
   */
  virtual void setIsMasterHasOutputFile(bool active) =0;
  virtual bool isMasterHasOutputFile() const =0;

  /*!
   * Sets the default verbosity level.
   *
   * Sets the verbosity levels for \a trace to the level \a minimal_level.
   * If the verbosity level is already higher than \a minimal_level, nothing
   * is done.
   * If \a minimal_level equals Arccore::Trace::UNSPECIFIED_VERBOSITY_LEVEL,
   * it resets the verbosity level to that specified by verbosityLevel()
   * and standardOutputVerbosityLevel().
   */
  virtual void setDefaultVerboseLevel(ITraceMng* trace,Int32 minimal_level) =0;

  virtual void setDefaultClassConfigXmlBuffer(ByteConstSpan bytes) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

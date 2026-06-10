// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CommandLineArguments.h                                      (C) 2000-2025 */
/*                                                                           */
/* Command Line Arguments.                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_COMMANDLINEARGUMENTS_H
#define ARCCORE_COMMON_COMMANDLINEARGUMENTS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounter.h"

#include "arccore/common/CommonGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ParameterList;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Command Line Arguments.
 *
 * This class uses a reference semantics.
 * The commandLineArgc() and commandLineArgv() methods return
 * pointers to internal structures of this class which are
 * allocated only as long as the instance is valid. They can be used
 * for classic C methods that expect pointers to the
 * command line arguments (i.e., the equivalent of the (argc,argv
 * pair of the main() function).
 *
 * Arguments starting with '-A' are considered as (key,value) parameters
 * type and must be in the form -A,x=y where `x` is the key and
 * `y` is the value. It is then possible to retrieve the value of a
 * parameter through its key using the getParameter() method;
 * If a parameter is present multiple times on the command line, the
 * last value is retained.
 */
class ARCCORE_COMMON_EXPORT CommandLineArguments
{
  class Impl;

 public:

  //! Create an instance from the arguments (argc,argv)
  CommandLineArguments(int* argc, char*** argv);
  CommandLineArguments();
  explicit CommandLineArguments(const StringList& args);
  CommandLineArguments(const CommandLineArguments& rhs);
  ~CommandLineArguments();
  CommandLineArguments& operator=(const CommandLineArguments& rhs);

 public:

  int* commandLineArgc() const;
  char*** commandLineArgv() const;

  //! Fills \a args with command line arguments.
  void fillArgs(StringList& args) const;

  /*!
   * \brief Retrieves the parameter with name \a param_name.
   *
   * Returns a null string if there is no parameter with this name.
   */
  String getParameter(const String& param_name) const;

  /*!
   * \brief Adds a parameter.
   * \sa ParameterList::addParameterLine()
   */
  void addParameterLine(const String& line);

  /*!
   * \brief Retrieves the list of parameters and their values.
   *
   * Returns the list of parameter names in \a param_names and
   * the associated value in \a values.
   */
  void fillParameters(StringList& param_names, StringList& values) const;

  //! List of parameters
  const ParameterList& parameters() const;

  //! List of parameters
  ParameterList& parameters();

  /*!
   * \brief Method to determine if the user requested
   * help on the command line.
   */
  bool needHelp() const;

 private:

  Arccore::ReferenceCounter<Impl> m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

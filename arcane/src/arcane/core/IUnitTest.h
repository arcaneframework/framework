// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IUnitTest.h                                                 (C) 2000-2024 */
/*                                                                           */
/* Interface of a unit test service.                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IUNITTEST_H
#define ARCANE_CORE_IUNITTEST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class XmlNode;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup StandardService
 * \brief Interface of a unit test service.
 */
class IUnitTest
{
 public:

  virtual ~IUnitTest() = default;

 public:

  //! Method called after reading the dataset but before reading the mesh
  virtual void buildInitializeTest() {}

  //! Method called after reading the dataset to initialize the test
  virtual void initializeTest() = 0;

  //! Method called to execute the test
  virtual void executeTest() = 0;

  //! Method called to release resources at the end of execution
  virtual void finalizeTest() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup StandardService
 * \brief Interface of a unit test service providing
 * a test report in the form of an XML node.
 */
class IXmlUnitTest
{
 public:

  virtual ~IXmlUnitTest() = default;

 public:

  virtual void buildInitializeTest() {}
  virtual void initializeTest() = 0;
  /*!
   * \brief Executes the test and fills the provided XML node parameter.
   *
   * Returns false for the code to stop in error, true otherwise
   * (useful for having an error in CMake tests...).
   */
  virtual bool executeTest(XmlNode& report) = 0;
  virtual void finalizeTest() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

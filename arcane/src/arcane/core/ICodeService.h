// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICodeService.h                                              (C) 2000-2025 */
/*                                                                           */
/* Interface of the code service.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ICODESERVICE_H
#define ARCANE_CORE_ICODESERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/IService.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a case loader.
 * \ingroup StandardService
 */
class ARCANE_CORE_EXPORT ICodeService
: public IService
{
 public:

  ~ICodeService() = default; //!< Frees resources

 public:

  /*! \brief Creates a session.
   *
   * The instance must call IApplication::addSession().
   */
  virtual ISession* createSession() = 0;

  /*!
   * \brief Parses the command line arguments.
   *
   * The array \a args only contains arguments that have not
   * been interpreted by Arcane.
   *
   * Recognized arguments must be removed from the list.
   *
   * \retval true if the execution must stop,
   * \retval false if it continues normally
   */
  virtual bool parseArgs(StringList& args) = 0;

  /*!
   * \brief Creates and loads the case using the info \a sdbi
   * for the session \a session.
   */
  virtual ISubDomain* createAndLoadCase(ISession* session, const SubDomainBuildInfo& sdbi) = 0;

  /*!
   * \brief Initializes the session \a session.
   *
   * \param is_continue indicates if we are resuming
   * The case must already have been loaded by loadCase()
   */
  virtual void initCase(ISubDomain* sub_domain, bool is_continue) = 0;

  //! Returns whether the code allows execution.
  virtual bool allowExecution() const = 0;

  /*! \brief Returns the list of file extensions processed by the instance.
   * The extension does not include the '.'.
   */
  virtual StringCollection validExtensions() const = 0;

  /*!
   * \brief Length unit used by the code.
   *
   * This must be 1.0 if the code uses the international system and thus
   * the meter as the length unit. If the unit is the centimeter, for
   * example, the value is 0.01.
   *
   * This value can be used, for example, when reading the
   * mesh if the mesh format supports the notion of length unit.
   */
  virtual Real ARCANE_DEPRECATED lengthUnit() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

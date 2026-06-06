// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IBase.h                                                     (C) 2000-2025 */
/*                                                                           */
/* Interface of a base object.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IBASE_H
#define ARCANE_CORE_IBASE_H
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
 * \brief Interface of the base class for main arcane objects
 */
class ARCANE_CORE_EXPORT IBase
{
 public:

  virtual ~IBase() = default; //!< Frees resources

 public:

  /*!
   * \brief Constructs the instance members.
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

 public:

  //! Parent of this object
  virtual IBase* objectParent() const = 0;

  //! Namespace of the object.
  virtual String objectNamespaceURI() const = 0;

  //! Local name of the object.
  virtual String objectLocalName() const = 0;

  //! Service version number.
  virtual VersionInfo objectVersion() const = 0;

 public:

  //! Trace manager
  virtual ITraceMng* traceMng() const = 0;

  //! Resource manager
  virtual IRessourceMng* ressourceMng() const = 0;

  //! Service manager
  virtual IServiceMng* serviceMng() const = 0;

 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CORE_EXPORT String
arcaneNamespaceURI();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

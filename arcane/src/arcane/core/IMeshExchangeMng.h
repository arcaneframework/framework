// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshExchangeMng.h                                          (C) 2000-2025 */
/*                                                                           */
/* Interface for managing mesh exchanges between subdomains.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHEXCHANGEMNG_H
#define ARCANE_CORE_IMESHEXCHANGEMNG_H
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
 * \brief Interface for managing mesh exchanges between
 * subdomains.
 */
class ARCANE_CORE_EXPORT IMeshExchangeMng
{
 public:

  virtual ~IMeshExchangeMng() = default; //!< Frees resources

 public:

  //! Associated mesh
  virtual IPrimaryMesh* mesh() const = 0;

  /*!
   * \brief Starts an exchange.
   *
   * \pre exchanger()==nullptr.
   *
   * While an exchange is in progress, certain
   * operations on the mesh are forbidden, such as creating a new family
   * or adding groups.
   */
  virtual IMeshExchanger* beginExchange() = 0;

  /*!
   * \brief Signals that the exchange is finished.
   *
   * This allows deallocating the structures associated with the exchange.
   * \post exchanger()==nullptr.
   */
  virtual void endExchange() = 0;

  /*!
   * \brief Current exchanger.
   *
   * The exchanger is non-null only if we are between a beginExchange() and an endExchange()
   */
  virtual IMeshExchanger* exchanger() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

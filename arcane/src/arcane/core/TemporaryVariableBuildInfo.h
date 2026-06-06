// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TemporaryVariableBuildInfo.h                                (C) 2000-2025 */
/*                                                                           */
/* Information for building a temporary variable.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_TEMPORARYVARIABLEBUILDINFO_H
#define ARCANE_CORE_TEMPORARYVARIABLEBUILDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/VariableBuildInfo.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IModule.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Parameters required to build a temporary variable.
 *
 * A variable, even a temporary one, must be created with the same parameters
 * on all sub-domains.
 *
 * \warning This class is not yet operational
 */
class ARCANE_CORE_EXPORT TemporaryVariableBuildInfo
: public VariableBuildInfo
{
 public:

  /*!
   * \brief Constructs an initializer for a variable.
   *
   * \param name name of the variable
   * \param m associated module
   */
  TemporaryVariableBuildInfo(IModule* m, const String& name);

  /*!
   * \brief Constructs an initializer for a variable without associating it with
   * a module.
   *
   * \param sub_domain sub-domain manager
   * \param name name of the variable
   */
  TemporaryVariableBuildInfo(ISubDomain* sub_domain, const String& name);

  /*!
   * \brief Constructs an initializer for a variable.
   *
   * \param m associated module
   * \param name name of the variable
   * \param item_family_name name of the entity family
   */
  TemporaryVariableBuildInfo(IModule* m, const String& name,
                             const String& item_family_name);

  /*!
   * \brief Constructs an initializer for a variable associated with a
   * mesh.
   *
   * \param sub_domain sub-domain manager
   * \param name name of the variable
   */
  TemporaryVariableBuildInfo(IMesh* mesh, const String& name);

  /*!
   * \brief Constructs an initializer for a variable associated with a
   * mesh.
   *
   * \param sub_domain sub-domain manager
   * \param name name of the variable
   * \param item_family_name name of the entity family
   */
  TemporaryVariableBuildInfo(IMesh* mesh, const String& name,
                             const String& item_family_name);

 protected:

  static int property();
  static String _generateName(IVariableMng* vm, const String& name);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

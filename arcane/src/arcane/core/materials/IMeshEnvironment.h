// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshEnvironment.h                                          (C) 2000-2023 */
/*                                                                           */
/* Interface of a mesh environment.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_IMESHENVIRONMENT_H
#define ARCANE_CORE_MATERIALS_IMESHENVIRONMENT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/IMeshComponent.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneMaterials
 * \brief Interface of a user environment.
 */
class ARCANE_CORE_EXPORT IUserMeshEnvironment
{
 public:

  virtual ~IUserMeshEnvironment() {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneMaterials
 * \brief Interface of a mesh environment.
 *
 * Environments are created via IMeshEnvironmentMng::createEnvironment().
 *
 * Environments cannot be destroyed, and all environments and their
 * materials must be created during initialization.
 *
 * An environment may optionally be empty.
 */
class ARCANE_CORE_EXPORT IMeshEnvironment
: public IMeshComponent
{
 public:

  virtual ~IMeshEnvironment() {}

 public:

  //! List of materials in this environment
  virtual ConstArrayView<IMeshMaterial*> materials() = 0;

  //! Number of materials in the environment
  virtual Integer nbMaterial() const = 0;

  /*!
   * \brief Environment identifier.
   * It is also the index (starting from 0) of this environment
   * in the list of environments.
   */
  //virtual Int32 id() const =0;

  //! Associated user environment
  virtual IUserMeshEnvironment* userEnvironment() const = 0;

  //! Sets the associated user environment
  virtual void setUserEnvironment(IUserMeshEnvironment* umm) = 0;

  /*!
   * \brief Cell of this environment for cell \a c.
   *
   * If this environment is not present in the cell,
   * a null environment cell is returned.
   *
   * The cost of this function is proportional to the number of materials
   * present in the cell.
   */
  virtual EnvCell findEnvCell(AllEnvCell c) const = 0;

  //! View associated with this environment
  virtual EnvItemVectorView envView() const = 0;

  //! View of the list of pure entities (associated with the global cell) in the environment
  virtual EnvPurePartItemVectorView pureEnvItems() const = 0;

  //! View of the list of impure (partial) entities in the environment
  virtual EnvImpurePartItemVectorView impureEnvItems() const = 0;

  //! View of the pure or impure part of the environment entities
  virtual EnvPartItemVectorView partEnvItems(eMatPart part) const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

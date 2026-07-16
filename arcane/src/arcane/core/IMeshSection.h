// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshSection.h                                              (C) 2000-2026 */
/*                                                                           */
/* Service interface allowing the creation of a mesh with a section of       */
/* another mesh.                                                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHSECTION_H
#define ARCANE_CORE_IMESHSECTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VariableCollection;
class MeshHandle;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Service interface allowing the creation of a mesh with a section of
 * another mesh.
 */
class ARCANE_CORE_EXPORT IMeshSection
{
 public:

  //! Releases resources
  virtual ~IMeshSection() = default;

 public:

  /*!
   * \brief Method allowing to add a plane to the cut service. The use of
   * these planes depends on the service.
   *
   * \param p0 Point of the plane.
   * \param normal Normal of the plane.
   */
  virtual void addPlane(const Real3& p0, const Real3& normal) = 0;

  /*!
   * \brief Method allowing to add a set of variables to copy on the
   * new mesh.
   *
   * \param variables A set of variable on the original mesh.
   */
  virtual void setVariables(VariableCollection variables) = 0;

  /*!
   * \brief Method allowing to get a set of variables copied on the new
   * mesh.
   *
   * \return A set of variables on the cloned mesh.
   */
  virtual VariableCollection variables() = 0;

  /*!
   * \brief Method allowing to update the mesh section with all planes.
   *
   * If a previous call has edited the mesh, all the cells will be destroyed
   * before update.
   */
  virtual void updateSection() = 0;

  /*!
   * \brief Méthod allowing to get the mesh section.
   *
   * \return The mesh section.
   */
  virtual MeshHandle meshSection() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDoFFamily.h                                                (C) 2000-2022 */
/*                                                                           */
/* Interface d'une famille de degrés de liberté (DoF).                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IDOFFAMILY_H
#define ARCANE_IDOFFAMILY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Interface d'une famille de DoF.
 */
class ARCANE_CORE_EXPORT IDoFFamily
{
 public:

  virtual ~IDoFFamily() = default; //<! Libère les ressources

 public:

  virtual void build() = 0;

 public:

  //! Nom de la famille
  virtual String name() const = 0;

  //! Nom complet de la famille (avec celui du maillage)
  virtual String fullName() const = 0;

  //! Nombre d'entités
  virtual Integer nbItem() const = 0;

  //! Groupe de tous les DoF
  virtual ItemGroup allItems() const = 0;

 public:

  //! En entree les uids des dofs et on recupere leurs lids
  virtual DoFVectorView addDoFs(Int64ConstArrayView dof_uids, Int32ArrayView dof_lids) = 0;

  //! L'ajout de fantomes doit etre suivi d'un appel de computeSynchronizeInfos
  virtual DoFVectorView addGhostDoFs(Int64ConstArrayView dof_uids, Int32ArrayView dof_lids,
                                     Int32ConstArrayView owners) = 0;

  virtual void removeDoFs(Int32ConstArrayView items_local_id) = 0;

  /*!
   * \sa IItemFamily::endUpdate().
   */
  virtual void endUpdate() = 0;

  virtual IItemFamily* itemFamily() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MatItem.h                                                   (C) 2000-2024 */
/*                                                                           */
/* Material and environment entities.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_MATITEM_H
#define ARCANE_CORE_MATERIALS_MATITEM_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/Item.h"

#include "arcane/core/materials/ComponentItem.h"
#include "arcane/core/materials/ComponentItemInternal.h"
#include "arcane/core/materials/IMeshMaterial.h"
#include "arcane/core/materials/IMeshEnvironment.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Represents a material in a multi-material mesh.
 *
 * This object represents a material in a multi-material mesh.
 *
 * There is a special mesh, called the null mesh, for which
 * null() is true and which represents an invalid mesh. In the
 * case of the invalid mesh, none of the other
 * class methods should be called, under penalty of causing a crash.
 *
 * \warning These meshes are invalidated as soon as the list of meshes of a
 * material or environment changes. Therefore, one must not
 * keep a mesh of this type between two changes of this list.
 */
class MatCell
: public ComponentCell
{
 public:

  ARCCORE_HOST_DEVICE MatCell(const matimpl::ConstituentItemBase& item_base)
  : ComponentCell(item_base)
  {
#ifdef ARCANE_CHECK
    _checkLevel(item_base, LEVEL_MATERIAL);
#endif
  }

  explicit ARCCORE_HOST_DEVICE MatCell(const ComponentCell& item)
  : MatCell(item.constituentItemBase())
  {
  }

  MatCell() = default;

 public:

  //! Environment mesh to which this material mesh belongs.
  ARCCORE_HOST_DEVICE inline EnvCell envCell() const;

  //! Associated material
  IMeshMaterial* material() const { return _material(); }

  //! Associated user material
  IUserMeshMaterial* userMaterial() const { return _material()->userMaterial(); }

  //! Material identifier
  ARCCORE_HOST_DEVICE Int32 materialId() const { return componentId(); }

 private:

  IMeshMaterial* _material() const { return static_cast<IMeshMaterial*>(component()); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Arcane mesh of an environment.
 *
 * Such a mesh contains information about the materials
 * of an environment for a given mesh.
 *
 * There is a special mesh, called the null mesh, for which
 * null() is true and which represents an invalid mesh. In the
 * case of the invalid mesh, none of the other
 * class methods should be called, under penalty of causing a crash.
 *
 * \warning These meshes are invalidated as soon as the list of meshes of a
 * material or environment changes. Therefore, one must not
 * keep a mesh of this type between two changes of this list.
 */
class EnvCell
: public ComponentCell
{
 public:

  explicit ARCCORE_HOST_DEVICE EnvCell(const matimpl::ConstituentItemBase& item_base)
  : ComponentCell(item_base)
  {
#ifdef ARCANE_CHECK
    _checkLevel(item_base, LEVEL_ENVIRONMENT);
#endif
  }
  explicit ARCCORE_HOST_DEVICE EnvCell(const ComponentCell& item)
  : EnvCell(item.constituentItemBase())
  {
  }
  EnvCell() = default;

 public:

  // Number of environment materials present in the mesh
  ARCCORE_HOST_DEVICE Int32 nbMaterial() const { return nbSubItem(); }

  //! Mesh containing information about all environments
  ARCCORE_HOST_DEVICE inline AllEnvCell allEnvCell() const;

  //! i-th material mesh of this mesh
  ARCCORE_HOST_DEVICE inline MatCell cell(Integer i) const { return _subItemBase(i); }

  //! Associated environment
  IMeshEnvironment* environment() const { return _environment(); }

  //! Environment identifier
  ARCCORE_HOST_DEVICE Int32 environmentId() const { return componentId(); }

  //! Enumerator over the material meshes of this mesh
  ARCCORE_HOST_DEVICE CellMatCellEnumerator subMatItems() const
  {
    return CellMatCellEnumerator(*this);
  }

 private:

  IMeshEnvironment* _environment() const { return static_cast<IMeshEnvironment*>(component()); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneMaterials
 * \brief Arcane mesh with material and environment information.
 *
 * Such a mesh contains information about the environments
 * for a given mesh. It allows, for example, knowing the number
 * of environments and for each the list of materials.
 *
 * \warning These meshes are invalidated as soon as the list of meshes of a
 * material or environment changes. Therefore, one must not
 * keep a mesh of this type between two changes of this list.
 */
class AllEnvCell
: public ComponentCell
{
 public:

  explicit ARCCORE_HOST_DEVICE AllEnvCell(const matimpl::ConstituentItemBase& item_base)
  : ComponentCell(item_base)
  {
#if defined(ARCANE_CHECK)
    _checkLevel(item_base, LEVEL_ALLENVIRONMENT);
#endif
  }

  explicit ARCCORE_HOST_DEVICE AllEnvCell(const ComponentCell& item)
  : AllEnvCell(item.constituentItemBase())
  {
  }

  AllEnvCell() = default;

 public:

  //! Number of environments present in the mesh
  ARCCORE_HOST_DEVICE Int32 nbEnvironment() const { return nbSubItem(); }

  //! i-th environment mesh
  EnvCell cell(Int32 i) const { return EnvCell(_subItemBase(i)); }

  //! Enumerator over the environment meshes of this mesh
  ARCCORE_HOST_DEVICE CellEnvCellEnumerator subEnvItems() const
  {
    return CellEnvCellEnumerator(*this);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCCORE_HOST_DEVICE inline EnvCell MatCell::
envCell() const
{
  return EnvCell(_superItemBase());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCCORE_HOST_DEVICE inline AllEnvCell EnvCell::
allEnvCell() const
{
  return AllEnvCell(_superItemBase());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

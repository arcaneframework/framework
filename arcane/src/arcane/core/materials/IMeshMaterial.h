// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshMaterial.h                                             (C) 2000-2023 */
/*                                                                           */
/* Interface of a mesh material.                                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_IMESHMATERIAL_H
#define ARCANE_CORE_MATERIALS_IMESHMATERIAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/IMeshComponent.h"

// This include is not useful for this '.h' but we keep it temporarily
// for compatibility with the existing code (June 2022).
#include "arcane/core/materials/IMeshMaterialMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneMaterials
 * \brief Interface of a user material.
 */
class ARCANE_CORE_EXPORT IUserMeshMaterial
{
 public:

  virtual ~IUserMeshMaterial(){}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneMaterials
 * \brief Interface of a mesh material.
 *
 * A material belongs to an environment (IMeshEnvironment). It is possible
 * to retrieve the list of meshes for this material via cells().
 */
class ARCANE_CORE_EXPORT IMeshMaterial
: public IMeshComponent
{
 public:

  virtual ~IMeshMaterial(){}

 public:

  //! Material information.
  virtual MeshMaterialInfo* infos() const =0;

  //! Environment to which this material belongs.
  virtual IMeshEnvironment* environment() const =0;

  //! Associated user material
  virtual IUserMeshMaterial* userMaterial() const =0;

  //! Sets the associated user material
  virtual void setUserMaterial(IUserMeshMaterial* umm) =0;

  /*!
   * \brief Mesh of this material for mesh \a c.
   *
   * If this material is not present in the mesh,
   * a null material mesh is returned.
   *
   * The cost of this function is proportional to the number of materials
   * present in the mesh.
   */   
  virtual MatCell findMatCell(AllEnvCell c) const =0;

  //! View associated with this material
  virtual MatItemVectorView matView() const =0;

  //! View on the list of pure entities (associated with the global mesh) of the material
  virtual MatPurePartItemVectorView pureMatItems() const =0;

  //! View on the list of impure (partial) entities of the material
  virtual MatImpurePartItemVectorView impureMatItems() const =0;

  //! View on the pure or impure part of the material entities
  virtual MatPartItemVectorView partMatItems(eMatPart part) const =0;

 public:

  void setImiInfo(Int32 first_imi,Int32 nb_imi)
  {
    m_first_imi = first_imi;
    m_nb_imi = nb_imi;
  }
  Int32 firstImi() const { return m_first_imi; }
  Int32 nbImi() const { return m_nb_imi; }

 protected:

  IMeshMaterial() : m_first_imi(-1), m_nb_imi(0){}

 private:

  Int32 m_first_imi;
  Int32 m_nb_imi;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

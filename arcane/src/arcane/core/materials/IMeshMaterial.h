// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshMaterial.h                                             (C) 2000-2022 */
/*                                                                           */
/* Interface d'un matériau d'un maillage.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_IMESHMATERIAL_H
#define ARCANE_CORE_MATERIALS_IMESHMATERIAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/IMeshComponent.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*class IMeshMaterialMng;
class MeshMaterialInfo;
class IMeshEnvironment;
class MeshMaterialVariableIndexer;
class MatItemVectorView;
class MatCell;
class AllEnvCell;*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Interface d'un matériau utilisateur.
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
 * \brief Interface d'un matériau d'un maillage.
 *
 * Un matériau appartient à un milieu (IMeshEnvironment). Il est possible
 * de récupérer la liste des mailles de ce matériau via cells().
 */
class ARCANE_CORE_EXPORT IMeshMaterial
: public IMeshComponent
{
 public:

  virtual ~IMeshMaterial(){}

 public:

  //! Infos du matériau.
  virtual MeshMaterialInfo* infos() const =0;

  //! Milieu auquel appartient ce matériau.
  virtual IMeshEnvironment* environment() const =0;

  //! Matériau utilisateur associé
  virtual IUserMeshMaterial* userMaterial() const =0;

  //! Positionne le matériau utilisateur associé
  virtual void setUserMaterial(IUserMeshMaterial* umm) =0;

  /*!
   * \brief Maille de ce matériau pour la maille \a c.
   *
   * Si ce matériau n'est pas présent dans la présent dans la maille,
   * la maille matériau nulle est retournée.
   *
   * Le coût de cette fonction est proportionnel au nombre de matériaux
   * présents dans la maille.
   */   
  virtual MatCell findMatCell(AllEnvCell c) const =0;

  //! Vue associée à ce matériau
  virtual MatItemVectorView matView() =0;

  //! Vue sur la liste des entités pures (associées à la maille globale) du matériau
  virtual MatPurePartItemVectorView pureMatItems() =0;

  //! Vue sur la liste des entités impures (partielles) partielles du matériau
  virtual MatImpurePartItemVectorView impureMatItems() =0;

  //! Vue sur la partie pure ou impure des entités du matériau
  virtual MatPartItemVectorView partMatItems(eMatPart part) =0;

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


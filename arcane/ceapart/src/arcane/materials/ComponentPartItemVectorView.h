// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentPartItemVectorView.h                               (C) 2000-2019 */
/*                                                                           */
/* Vue sur un vecteur sur une partie des entités composants.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_COMPONENTPARTITEMVECTORVIEW_H
#define ARCANE_MATERIALS_COMPONENTPARTITEMVECTORVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

#include "arcane/materials/MaterialsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Vue sur une partie pure ou partielles des entités d'un composant.
 */
class ARCANE_MATERIALS_EXPORT ComponentPartItemVectorView
{
 public:

  /*!
   * \brief Construit une vue sur une partie des entité du composant \a component.
   *
   * Ce constructeur n'est en principe pas appelé directement. Pour construire
   * une telle vue il est préférable de passer par les méthodes
   * IMeshComponent::pureItems(), IMeshComponent::impureItems() ou
   * IMeshComponent::partItems().
   */
  ComponentPartItemVectorView(IMeshComponent* component,Int32 component_part_index,
                              Int32ConstArrayView value_indexes,
                              Int32ConstArrayView item_indexes,
                              ConstArrayView<ComponentItemInternal*> items_internal,
                              eMatPart part)
  : m_component(component), m_component_part_index(component_part_index),
    m_value_indexes(value_indexes), m_item_indexes(item_indexes),
    m_items_internal(items_internal), m_part(part)
  {
  }
 public:
  //! Construit une vue non initialisée
  ComponentPartItemVectorView()
  : m_component(nullptr), m_component_part_index(-1)
  {
  }

 public:

  //! Nombre d'entités dans la vue
  Integer nbItem() const { return m_value_indexes.size(); }

  //! Composant associé
  IMeshComponent* component() const { return m_component; }

  // Index de la partie de ce composant (équivalent à MatVarIndex::arrayIndex()).
  Int32 componentPartIndex() const { return m_component_part_index; }

    //! Liste des valueIndex() de la partie
  Int32ConstArrayView valueIndexes() const { return m_value_indexes; }

  //! Liste des indices dans \a itemsInternal() des entités.
  Int32ConstArrayView itemIndexes() const { return m_item_indexes; }

  //! Tableau parties internes des entités
  ConstArrayView<ComponentItemInternal*> itemsInternal() const { return m_items_internal; }

  //! Partie du composant.
  eMatPart part() const { return m_part; }

 private:

  //! Gestionnaire de constituants
  IMeshComponent* m_component;

  //! Indice du constituant pour l'accès aux valeurs partielles.
  Int32 m_component_part_index;

  //! Liste des valueIndex() de la partie
  Int32ConstArrayView m_value_indexes;

  //! Liste des indices dans \a m_items_internal de chaque maille matériau.
  Int32ConstArrayView m_item_indexes;

  //! Liste des ComponentItemInternal* pour ce constituant.
  ConstArrayView<ComponentItemInternal*> m_items_internal;

  //! Partie du constituant
  eMatPart m_part;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Vue sur la partie pure d'un composant.
 */
class ARCANE_MATERIALS_EXPORT ComponentPurePartItemVectorView
: public ComponentPartItemVectorView
{
 public:
  //! Construit une vue sur une partie des entité du composant \a component.
  ComponentPurePartItemVectorView(IMeshComponent* component,
                                  Int32ConstArrayView value_indexes,
                                  Int32ConstArrayView item_indexes,
                                  ConstArrayView<ComponentItemInternal*> items_internal)
  : ComponentPartItemVectorView(component,0,value_indexes,item_indexes,items_internal,eMatPart::Pure)
  {
  }
  //! Construit une vue non initialisée
  ComponentPurePartItemVectorView() {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Vue sur la partie impure d'un composant.
 */
class ARCANE_MATERIALS_EXPORT ComponentImpurePartItemVectorView
: public ComponentPartItemVectorView
{
 public:
  //! Construit une vue sur une partie des entité du composant \a component.
  ComponentImpurePartItemVectorView(IMeshComponent* component,
                                    Int32 component_part_index,
                                    Int32ConstArrayView value_indexes,
                                    Int32ConstArrayView item_indexes,
                                    ConstArrayView<ComponentItemInternal*> items_internal)
  : ComponentPartItemVectorView(component,component_part_index,value_indexes,
                                item_indexes,items_internal,eMatPart::Impure)
  {
  }
  //! Construit une vue non initialisée
  ComponentImpurePartItemVectorView() {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Vue sur une partie pure ou partielles des entités d'un matériau.
 */
class ARCANE_MATERIALS_EXPORT MatPartItemVectorView
: public ComponentPartItemVectorView
{
 public:

  //! Construit une vue pour le matériau \a material.
  MatPartItemVectorView(IMeshMaterial* material,const ComponentPartItemVectorView& view);
  //! Construit une vue non initialisée
  MatPartItemVectorView()
  : m_material(nullptr) { }

 public:

  //! Matériau associé
  IMeshMaterial* material() const { return m_material; }

 private:

  IMeshMaterial* m_material;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Vue sur la partie pure des entités d'un matériau.
 */
class ARCANE_MATERIALS_EXPORT MatPurePartItemVectorView
: public MatPartItemVectorView
{
 public:

  //! Construit une vue pour le matériau \a material.
  MatPurePartItemVectorView(IMeshMaterial* material,const ComponentPurePartItemVectorView& view)
  : MatPartItemVectorView(material,view) {}
  //! Construit une vue non initialisée
  MatPurePartItemVectorView() : MatPartItemVectorView(){}

 public:

  operator ComponentPurePartItemVectorView() const
  {
    return ComponentPurePartItemVectorView(component(),valueIndexes(),
                                           itemIndexes(),itemsInternal());
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Vue sur la partie impure des entités d'un matériau.
 */
class ARCANE_MATERIALS_EXPORT MatImpurePartItemVectorView
: public MatPartItemVectorView
{
 public:

  //! Construit une vue pour le matériau \a material.
  MatImpurePartItemVectorView(IMeshMaterial* material,const ComponentImpurePartItemVectorView& view)
  : MatPartItemVectorView(material,view) {}
  //! Construit une vue non initialisée
  MatImpurePartItemVectorView() : MatPartItemVectorView(){}

 public:

  operator ComponentImpurePartItemVectorView() const
  {
    return ComponentImpurePartItemVectorView(component(),componentPartIndex(),
                                             valueIndexes(),
                                             itemIndexes(),itemsInternal());
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Vue sur une partie pure ou partielles des entités d'un milieu.
 */
class ARCANE_MATERIALS_EXPORT EnvPartItemVectorView
: public ComponentPartItemVectorView
{
 public:

  //! Construit une vue pour le milieu \a env.
  EnvPartItemVectorView(IMeshEnvironment* env,const ComponentPartItemVectorView& view);
  //! Construit une vue non initialisée
  EnvPartItemVectorView()
  : m_environment(nullptr) { }

 public:

  //! Matériau associé
  IMeshEnvironment* environment() const { return m_environment; }

 private:

  IMeshEnvironment* m_environment;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Vue sur la partie pure des entités d'un milieu.
 */
class ARCANE_MATERIALS_EXPORT EnvPurePartItemVectorView
: public EnvPartItemVectorView
{
 public:

  //! Construit une vue pour le milieu \a env.
  EnvPurePartItemVectorView(IMeshEnvironment* env,const ComponentPurePartItemVectorView& view)
  : EnvPartItemVectorView(env,view) {}
  //! Construit une vue non initialisée
  EnvPurePartItemVectorView() : EnvPartItemVectorView(){}

 public:

  operator ComponentPurePartItemVectorView() const
  {
    return ComponentPurePartItemVectorView(component(),valueIndexes(),
                                           itemIndexes(),itemsInternal());
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Vue sur la partie impure des entités d'un milieu.
 */
class ARCANE_MATERIALS_EXPORT EnvImpurePartItemVectorView
: public EnvPartItemVectorView
{
 public:

  //! Construit une vue pour le milieu \a env.
  EnvImpurePartItemVectorView(IMeshEnvironment* env,const ComponentImpurePartItemVectorView& view)
  : EnvPartItemVectorView(env,view) {}
  //! Construit une vue non initialisée
  EnvImpurePartItemVectorView() : EnvPartItemVectorView(){}

 public:

  operator ComponentImpurePartItemVectorView() const
  {
    return ComponentImpurePartItemVectorView(component(),componentPartIndex(),
                                             valueIndexes(),
                                             itemIndexes(),itemsInternal());
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

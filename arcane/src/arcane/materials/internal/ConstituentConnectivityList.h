// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConstituentConnectivityList.h                               (C) 2000-2024 */
/*                                                                           */
/* Gestion des listes de connectivité des constituants.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_COMPONENTCONNECTIVITYLIST_H
#define ARCANE_MATERIALS_INTERNAL_COMPONENTCONNECTIVITYLIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/Ref.h"
#include "arcane/utils/DualUniqueArray.h"

#include "arcane/core/VariableTypes.h"
#include "arcane/core/IIncrementalItemConnectivity.h"

#include "arcane/materials/MaterialsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Gestion des listes de connectivité des constituants.
 */
class ConstituentConnectivityList
: public TraceAccessor
, public ReferenceCounterImpl
, public IIncrementalItemSourceConnectivity
{
  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();

  class ConstituentContainer;
  class Container;
  class NumberOfMaterialComputer;

 public:

  explicit ConstituentConnectivityList(MeshMaterialMng* mm);
  ~ConstituentConnectivityList();

 public:

  ConstituentConnectivityList(ConstituentConnectivityList&&) = delete;
  ConstituentConnectivityList(const ConstituentConnectivityList&) = delete;
  ConstituentConnectivityList& operator=(ConstituentConnectivityList&&) = delete;
  ConstituentConnectivityList& operator=(const ConstituentConnectivityList&) = delete;

 public:

  void endCreate(bool is_continue);

  void addCellsToEnvironment(Int16 env_id, SmallSpan<const Int32> cell_ids, RunQueue& queue);
  void removeCellsToEnvironment(Int16 env_id, SmallSpan<const Int32> cell_ids, RunQueue& queue);

  void addCellsToMaterial(Int16 mat_id, SmallSpan<const Int32> cell_ids, RunQueue& queue);
  void removeCellsToMaterial(Int16 mat_id, SmallSpan<const Int32> cell_ids, RunQueue& queue);

  //! Tableaux du nombre total de milieux par maille (indexé par localId())
  ConstArrayView<Int16> cellsNbEnvironment() const;
  //! Tableaux du nombre total de matériaux par maille (indexé par localId())
  ConstArrayView<Int16> cellsNbMaterial() const;

  //! Nombre de matériaux de la maille \a cell_id pour le milieu d'indice \a env_id
  Int16 cellNbMaterial(CellLocalId cell_id, Int16 env_id) const;

  //! Supprime toutes les entités connectées
  void removeAllConnectivities();

  /*!
   * \brief Remplit \a cells_nb_material avec le nombre de matériaux du milieu \a env_id
   */
  void fillCellsNbMaterial(SmallSpan<const Int32> cells_local_id, Int16 env_id,
                           SmallSpan<Int16> cells_nb_material, RunQueue& queue);

  /*!
   * \brief Replit \a cells_do_transform en indiquant is la maille passe de pure à partielle.
   */
  void fillCellsToTransform(SmallSpan<const Int32> cells_local_id, Int16 env_id,
                            SmallSpan<bool> cells_do_transform, bool is_add, RunQueue& queue);

  /*!
   * \brief Replit \a cells_is_partial en indiquant is la maille est partielle pour le milieu \a env_id
   */
  void fillCellsIsPartial(SmallSpan<const Int32> cells_local_id, Int16 env_id,
                          SmallSpan<bool> cells_is_partial, RunQueue& queue);

  /*!
   * \brief Indique si l'instance est activée.
   *
   * Valide uniquement après appel à endCreate().
   * Si l'instance n'est pas active, il ne faut appeler aucune des méthodes qui
   * modifient la liste des constituants.
   */
  bool isActive() const { return m_is_active; }

 public:

  // Implémentation de IIncrementalItemSourceConnectivity
  //@{
  IItemFamily* sourceFamily() const override { return m_cell_family; }
  void notifySourceFamilyLocalIdChanged(Int32ConstArrayView new_to_old_ids) override;
  void notifySourceItemAdded(ItemLocalId item) override;
  void reserveMemoryForNbSourceItems(Int32 n, bool pre_alloc_connectivity) override;
  void notifyReadFromDump() override;
  Ref<IIncrementalItemSourceConnectivity> toSourceReference() override;
  //@}

 private:

  MeshMaterialMng* m_material_mng = nullptr;
  IItemFamily* m_cell_family = nullptr;
  Container* m_container = nullptr;

  //! Indice du milieu auquel appartient un matériau
  DualUniqueArray<Int16> m_environment_for_materials;
  bool m_is_active = false;

 public:

  void _addCells(Int16 env_id, SmallSpan<const Int32> cell_ids,
                 ConstituentContainer& component, RunQueue& queue);
  void _removeCells(Int16 env_id, SmallSpan<const Int32> cell_ids,
                    ConstituentContainer& component, RunQueue& queue);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AllCellToAllEnvCellContainer.h                              (C) 2000-2024 */
/*                                                                           */
/* Conteneur des données pour 'AllCellToAllEnvCell'.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_ALLCELLTOALLENVCELLCONTAINER_H
#define ARCANE_MATERIALS_INTERNAL_ALLCELLTOALLENVCELLCONTAINER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/MaterialsGlobal.h"

#include "arcane/utils/NumArray.h"

#include "arcane/materials/AllCellToAllEnvCellConverter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
class MeshMaterialAcceleratorUnitTest;
}

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Table de connectivité des 'Cell' vers leur(s) 'AllEnvCell' destinée
 *        à une utilisation sur accélérateur.
 *
 * Classe qui conserve la connectivité de toutes les mailles
 * \a Cell vers toutes leurs mailles \a AllEnvCell.
 *
 * On crée une instance via la méthode create().
 *
 * Le coût de l'initialisation est cher, il faut allouer la mémoire et remplir les
 * structures. On parcours toutes les mailles et pour chaque maille on fait
 * appel au CellToAllEnvCellConverter.
 *
 * Une fois l'instance créée, elle doit être mise à jour à chaque fois que
 * la topologie des matériaux/environnements change (ce qui est également cher).
 *
 * Cette classe est une classe interne et ne doit pas être manipulée directement.
 * Il faut passer par les helpers associés dans le IMeshMaterialMng et
 * la classe CellToAllEnvCellAccessor.
 */
class ARCANE_MATERIALS_EXPORT AllCellToAllEnvCellContainer
: public AllCellToAllEnvCell
{
 public:

  class Impl;

 public:

  explicit AllCellToAllEnvCellContainer(IMeshMaterialMng* mm);

 public:

  //! Copies interdites
  AllCellToAllEnvCellContainer& operator=(const AllCellToAllEnvCellContainer&) = delete;

 public:

  /*!
   * \brief Fonction de création alternative. Il faut attendre que les données
   * relatives aux matériaux soient finalisées.
   *
   * La différence réside dans la gestion de la mémoire.
   * Ici, on applique un compromis sur la taille de la table cid -> envcells
   * où la taille du tableau pour ranger les envcells d'une cell est égale à la taille
   * max du nb d'environnement présent à un instant t dans un maille.
   * Celà permet de ne pas faire les allocations mémoire dans la boucle interne et de
   * façon systématique.
   * => Gain de perf à évaluer.
   */
  void initialize();

  //! Méthode d'accès à la table de "connectivité" cell -> all env cells
  ARCCORE_HOST_DEVICE Span<ComponentItemLocalId>* internal() const
  {
    return m_allcell_allenvcell_ptr;
  }

  /*!
   * \brief Méthode pour donner le nombre maximal d'environnements
   * présents sur une maille à l'instant t.
   *
   * Le fait d'effectuer cette opération à un instant donné, permet
   * d'avoir une valeur max <= au nombre total d'environnement présents
   * dans le jdd (et donc d'économiser un peu de mémoire).
   */
  Int32 maxNbEnvPerCell() const;

  /*!
   * On regarde si le nb max d'env par cell à l'instant t a changé,
   * et si c'est le cas, on force la reconstruction de la table.
   * Est appelé par le forceRecompute du IMeshMaterialMng
   */
  void bruteForceUpdate();

  void reset();

 private:

  IMeshMaterialMng* m_material_mng = nullptr;
  Integer m_size = 0;
  NumArray<Span<ComponentItemLocalId>, MDDim1> m_allcell_allenvcell;
  //Span<ComponentItemLocalId>* m_allcell_allenvcell_ptr = nullptr;
  NumArray<ComponentItemLocalId, MDDim1> m_mem_pool;
  Int32 m_current_max_nb_env = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  


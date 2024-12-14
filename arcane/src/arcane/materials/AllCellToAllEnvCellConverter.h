// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AllCellToAllEnvCellConverter.h                              (C) 2000-2024 */
/*                                                                           */
/* Conversion de 'Cell' en 'AllEnvCell'.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_ALLCELLTOALLENVCELLCONVERTER_H
#define ARCANE_MATERIALS_ALLCELLTOALLENVCELLCONVERTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/MaterialsGlobal.h"

#include "arcane/core/materials/MatItem.h"
#include "arcane/core/materials/MatItemEnumerator.h"
#include "arcane/core/materials/CellToAllEnvCellConverter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
class ARCANE_MATERIALS_EXPORT AllCellToAllEnvCell
{
  friend class CellToAllEnvCellAccessor;
  friend class CellToAllComponentCellEnumerator;
  friend AllCellToAllEnvCellContainer;

 private:

  //! Méthode d'accès à la table de "connectivité" cell -> all env cells
  ARCCORE_HOST_DEVICE Span<ComponentItemLocalId>* _internal() const
  {
    return m_allcell_allenvcell_ptr;
  }

 private:

  Span<ComponentItemLocalId>* m_allcell_allenvcell_ptr = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Classe d'encapsulation pour accéder à la connectivité équivalente 
 *        cell -> allenvcell. Destinée à être utilisée avec l'API accélérateur
 *        via les RUNCOMMAND_...
 * \note Aucun interet en soit, mis à part le fait d'obliger l'utilisateur à créer
 * cet objet en amout de l'appel à un RUNCOMMAND_ENUMERATE_CELL_ALLENVCELL et donc 
 * de garantir la copie du pointeur AllCellToAllEnvCell pour la lambda à executer sur
 * l'accélérateur
 */
class ARCANE_MATERIALS_EXPORT CellToAllEnvCellAccessor
{
  friend class CellToAllComponentCellEnumerator;

 public:

  using size_type = Span<ComponentItemLocalId>::size_type;

 public:

  CellToAllEnvCellAccessor() = default;
  explicit CellToAllEnvCellAccessor(const IMeshMaterialMng* mm);

  ARCCORE_HOST_DEVICE size_type nbEnvironment(Int32 cid) const
  {
    return m_cell_allenvcell._internal()[cid].size();
  }

 private:

  ARCCORE_HOST_DEVICE const AllCellToAllEnvCell* _getAllCellToAllEnvCell() const
  {
    return &m_cell_allenvcell;
  }

 private:

  AllCellToAllEnvCell m_cell_allenvcell;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_MATERIALS_EXPORT CellToAllComponentCellEnumerator
{
  friend class EnumeratorTracer;

 public:

  using index_type = Span<ComponentItemLocalId>::index_type;
  using size_type = Span<ComponentItemLocalId>::size_type;

 public:

  // La version CPU permet de vérifier qu'on a bien fait l'init avant l'ENUMERATE
  ARCCORE_HOST_DEVICE explicit CellToAllComponentCellEnumerator(Int32 cell_id, const CellToAllEnvCellAccessor& acc)
  : m_cid(cell_id)
  {
    const AllCellToAllEnvCell* all_env_ptr = acc._getAllCellToAllEnvCell();
#if defined(ARCCORE_DEVICE_CODE)
    m_ptr = &(all_env_ptr->_internal()[cell_id]);
    m_size = m_ptr->size();
#else
    if (all_env_ptr) {
      m_ptr = &(all_env_ptr->_internal()[cell_id]);
      m_size = m_ptr->size();
    }
    else
      ARCANE_FATAL("Must create AllCellToAllEnvCell before using ENUMERATE_ALLENVCELL");
#endif
  }
  ARCCORE_HOST_DEVICE void operator++()
  {
    ++m_index;
  }

  ARCCORE_HOST_DEVICE bool hasNext() const
  {
    return m_index < m_size;
  }

  ARCCORE_HOST_DEVICE ComponentItemLocalId operator*() const
  {
    return (*m_ptr)[m_index];
  }

 private:

  Int32 m_cid = 0;
  index_type m_index = 0;
  Span<ComponentItemLocalId>* m_ptr = nullptr;
  size_type m_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief  Macro pour itérer sur un groupe de mailles dans le but d'itérer
 * sur les allenvcell de chaque maille.
 *
 * \note En forçant l'utilisation du CellToAllEnvCellAccessor dans la macro,
 * on assure la capture par copie du pointeur de AllCellToAllEnvCell, permettant
 * l'utilisation du ENUMERATE_CELL_ALLENVCELL.
 *
 * TODO Très certainement à déplacer ailleurs si on garde ce proto
 */
#define RUNCOMMAND_ENUMERATE_CELL_ALLENVCELL(cell_to_allenvcellaccessor, iter_name, cell_group) \
  A_FUNCINFO << cell_group << [=] ARCCORE_HOST_DEVICE(CellLocalId iter_name)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: Très certainement à déplacer ailleurs si on garde ce proto
#define A_ENUMERATE_CELL_ALLCOMPONENTCELL(_EnumeratorClassName, iname, cid, cell_to_allenvcellaccessor) \
  for (A_TRACE_COMPONENT(_EnumeratorClassName) iname(::Arcane::Materials::_EnumeratorClassName(cid, cell_to_allenvcellaccessor) A_TRACE_ENUMERATOR_WHERE); iname.hasNext(); ++iname)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Macro pour itérer sur tous les milieux d'une maille à l'intérieur.
 *        Version "brute et légère" ENUMERATE_CELL_ENVCELL, destinée à un 
 *        emploi sur accélérateur, i.e. au sein d'un RUN_COMMAND...
 *
 * \param iname nom de la variable (type MatVarIndex) permettant l'accès aux 
 *              données.
 * \param cid identifiant de la maille (type Integer).
 * \param cell_to_allenvcellaccessor connectivité cell->allenvcell (type CellToAllEnvCellAccessor)
 */
// TODO: Très certainement à déplacer ailleurs si on garde ce proto
#define ENUMERATE_CELL_ALLENVCELL(iname, cid, cell_to_allenvcellaccessor) \
  A_ENUMERATE_CELL_ALLCOMPONENTCELL(CellToAllComponentCellEnumerator, iname, cid, cell_to_allenvcellaccessor)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  


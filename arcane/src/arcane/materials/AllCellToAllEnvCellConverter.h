// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AllCellToAllEnvCellConverter.h                              (C) 2000-2023 */
/*                                                                           */
/* Conversion de 'Cell' en 'AllEnvCell'.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_ALLCELLTOALLENVCELLCONVERTER_H
#define ARCANE_CORE_MATERIALS_ALLCELLTOALLENVCELLCONVERTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/MaterialsGlobal.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/materials/MatItem.h"
#include "arcane/core/materials/IMeshMaterialMng.h"
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

 Classe qui stoque la connectivité de toutes les mailles 
 \a Cell vers toutes leurs mailles \a AllEnvCell.
 
 On crée une instance via la méthode create().

 Le coût de l'initialisation est cher, il faut allouer la mémoire et remplir les 
 structures. On parcours toutes les mailles et pour chaque maille on fait 
 appel au CellToAllEnvCellConverter.

 Une fois l'instance créée, elle doit être mise à jour à chaque fois que 
 la topologie des matériaux/environnements change (ce qui est également cher).
 
 \code
 * 
 * auto* ac2aec = ::Arcane::Materials::AllCell2AllEnvCell::create(...);
 * command << RUNCOMMAND_ENUMERATE(Cell, cid, allCells()) {
 *   for (Int32 i(0); i < ac2aec->getAllCell2AllEnvCellTable()[cid].size(); ++i) {
 *     const auto& mvi(ac2aec->getAllCell2AllEnvCellTable()[cid][i]);
 *     sum2 += in_menv_var2[mvi]/in_menv_var2_g[cid];
 *     ...
 *   }
 * };
 \endcode
 
 */
class ARCANE_MATERIALS_EXPORT AllCell2AllEnvCell
{
 public:
  //AllCell2AllEnvCell();
  //~AllCell2AllEnvCell();
  void reset();
  
  //! Copies interdites
  AllCell2AllEnvCell(const AllCell2AllEnvCell&) = delete;
  AllCell2AllEnvCell& operator=(const AllCell2AllEnvCell&) = delete;

  /*!
   * La fonction de création. Il faut attendre que les données
   * relatives aux matériaux soient finalisées
   */
  static AllCell2AllEnvCell* create(IMeshMaterialMng* mm, IMemoryAllocator* alloc);

  // Rien d'intelligent ici, on refait tout. Il faut voir dans un 2nd temps 
  // pour faire qqch de plus malin et donc certainement plus rapide...
  // SI on garde cette classe et ce concept... ça m'étonnerait...
  void bruteForceUpdate(Int32ConstArrayView ids);

  /*!
   * Méthode d'accès à la table de "connectivité" cell -> all env cells
   */
  ARCCORE_HOST_DEVICE Span<ComponentItemLocalId>* internal() const
  {
    return m_allcell_allenvcell;
  }

 private:
  IMeshMaterialMng* m_mm;
  IMemoryAllocator* m_alloc;
  Integer m_nb_allcell;
  Span<ComponentItemLocalId>* m_allcell_allenvcell;
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
 * de garantir la copie du pointeur AllCell2AllEnvCell pour la lambda à executer sur
 * l'accélérateur
 */
class ARCANE_MATERIALS_EXPORT Cell2AllEnvCellAccessor
{
 public:
  Cell2AllEnvCellAccessor()
  : m_cell_allenvcell(nullptr)
  {
  }
  Cell2AllEnvCellAccessor(const IMeshMaterialMng* mmmng)
  : m_cell_allenvcell(mmmng->getAllCell2AllEnvCell())
  {
  }
  ARCCORE_HOST_DEVICE Cell2AllEnvCellAccessor(const Cell2AllEnvCellAccessor& acc)
  : m_cell_allenvcell(acc.m_cell_allenvcell)
  {
  }
  ARCCORE_HOST_DEVICE Cell2AllEnvCellAccessor(Cell2AllEnvCellAccessor& acc)
  : m_cell_allenvcell(acc.m_cell_allenvcell)
  {
  }

  ARCCORE_HOST_DEVICE Cell2AllEnvCellAccessor& operator=(Cell2AllEnvCellAccessor acc)
  {
    m_cell_allenvcell = acc.m_cell_allenvcell;
    return *this;
  }
  ARCCORE_HOST_DEVICE Cell2AllEnvCellAccessor& operator=(const Cell2AllEnvCellAccessor& acc)
  {
    m_cell_allenvcell = acc.m_cell_allenvcell;
    return *this;
  }

  ARCCORE_HOST_DEVICE AllCell2AllEnvCell* getAllCell2AllEnvCell() const
  {
    return m_cell_allenvcell;
  }

 private:
  AllCell2AllEnvCell* m_cell_allenvcell;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
class ARCANE_MATERIALS_EXPORT Cell2AllComponentCellEnumerator
{
  friend class EnumeratorTracer;

 public:
  using index_type = Span<ComponentItemLocalId>::index_type;
  using size_type = Span<ComponentItemLocalId>::size_type;

 public:
  // La version CPU permet de vérifier qu'on a bien fait l'init avant l'ENUMERATE
  ARCCORE_HOST_DEVICE explicit Cell2AllComponentCellEnumerator(Integer cell_id, const Cell2AllEnvCellAccessor& acc)
  : m_cid(cell_id), m_index(0)
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  , m_ptr(&(acc.getAllCell2AllEnvCell()->internal()[cell_id]))
  , m_size((*m_ptr).size())
  {
  }
#else
  , m_ptr(nullptr), m_size(0)
  {
    if (acc.getAllCell2AllEnvCell()) {
      m_ptr = &(acc.getAllCell2AllEnvCell()->internal()[cell_id]);
      m_size = (*m_ptr).size();
    } else {
      ARCANE_FATAL("Must create AllCell2AllEnvCell before using ENUMERATE_ALLENVCELL");
    }
  }
#endif

  ARCCORE_HOST_DEVICE void operator++() { ++m_index; }

  ARCCORE_HOST_DEVICE bool hasNext() const { return m_index<m_size; }

  ARCCORE_HOST_DEVICE ComponentItemLocalId& operator*() const { return (*m_ptr)[m_index]; }

 private:
  Integer m_cid;
  index_type m_index;
  Span<ComponentItemLocalId>* m_ptr;
  size_type m_size;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Macro pour itérer sur un groupe de mailles dans le but d'itérer sur les allenvcell de chaque maille
//!\note En forçant l'utilisation du Cell2AllEnvCellAccessor dans la macro, on assure la capture par copie
// du pointeur de AllCell2AllEnvCell, permettant l'utilisation du ENUMERATE_CELL_ALLENVCELL
// TODO: Très certainement à déplacer ailleurs si on garde ce proto
#define RUNCOMMAND_ENUMERATE_CELL_ALLENVCELL(cell2allenvcellaccessor,iter_name,cell_group)         \
  A_FUNCINFO << cell_group << [=] ARCCORE_HOST_DEVICE (CellLocalId iter_name)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: Très certainement à déplacer ailleurs si on garde ce proto
#define A_ENUMERATE_CELL_ALLCOMPONENTCELL(_EnumeratorClassName,iname,cid,cell2allenvcellaccessor) \
  for( A_TRACE_COMPONENT(_EnumeratorClassName) iname(::Arcane::Materials::_EnumeratorClassName(cid,cell2allenvcellaccessor) A_TRACE_ENUMERATOR_WHERE); iname.hasNext(); ++iname )

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
 * \param cell2allenvcellaccessor connectivité cell->allenvcell (type Cell2AllEnvCellAccessor)
 */
// TODO: Très certainement à déplacer ailleurs si on garde ce proto
#define ENUMERATE_CELL_ALLENVCELL(iname,cid,cell2allenvcellaccessor) \
  A_ENUMERATE_CELL_ALLCOMPONENTCELL(Cell2AllComponentCellEnumerator,iname,cid,cell2allenvcellaccessor)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

